import os
import pandas as pd
import time
import re
import argparse
import itertools
from tqdm import tqdm

import os
import sys
import argparse
from typing import List
import shutil

import openai
import time

import csv

# import env
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

from webagents_step.utils.data_prep import *
from webagents_step.agents.prompt_agent import PromptAgent
from webagents_step.agents.step_agent import StepAgent
from webagents_step.prompts.webarena import flat_fewshot_template, step_fewshot_template
from webagents_step.environment.webarena import WebArenaEnvironmentWrapper

openai.api_key = os.environ.get("OPENAI_API_KEY")

def run():
    # parser = argparse.ArgumentParser(
    #     description="Only the config file argument should be passed"
    # )
    # parser.add_argument(
    #     "--config", type=str, required=True, help="yaml config file location"
    # )
    # args = parser.parse_args()

    config_path = "./step_settings.yml"

    with open(config_path, "r") as file:
        config = DotDict(yaml.safe_load(file))


    # setting root agent
    config.agent.root_action = 'github_agent'
    
    dstdir = f"{config.logdir}/{time.strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(dstdir, exist_ok=True)
    shutil.copyfile(config_path, os.path.join(dstdir, config_path.split("/")[-1]))
    random.seed(42)
    
    config_file_list = []
    
    # task_ids = [44, 45, 156]
    task_ids = config.env.task_ids

    for task_id in task_ids:
        config_file_list.append(f"tasks/webarena/{task_id}.json")

    action_to_prompt_dict = {k: v for k, v in step_fewshot_template.__dict__.items() if isinstance(v, dict)}
    low_level_action_list = config.agent.low_level_action_list

    if config.agent.type == "step":
        agent_init = lambda: StepAgent(
        root_action = config.agent.root_action,
        action_to_prompt_dict = action_to_prompt_dict,
        low_level_action_list = low_level_action_list,
        max_actions=config.env.max_env_steps,
        verbose=config.verbose,
        logging=config.logging,
        debug=config.debug,
        model=config.agent.model_name,
        prompt_mode=config.agent.prompt_mode,
        )
    elif config.agent.type == "flat_fewshot8k":
        agent_init = lambda: PromptAgent(
            prompt_template=flat_fewshot_template.flat_fewshot_agent8k,
            model=config.agent.model_name,
            prompt_mode=config.agent.prompt_mode,
            max_actions=config.env.max_env_steps,
            verbose=config.verbose,
            logging=config.logging,
            debug=config.debug,
        )
    elif config.agent.type == "flat_fewshot4k":
        agent_init = lambda: PromptAgent(
            prompt_template=flat_fewshot_template.flat_fewshot_agent4k,
            model=config.agent.model_name,
            prompt_mode=config.agent.prompt_mode,
            max_actions=config.env.max_env_steps,
            verbose=config.verbose,
            logging=config.logging,
            debug=config.debug,
        )
    else:
        raise NotImplementedError(f"{config.agent.type} not implemented")

    #####
    # Evaluate
    #####

    for config_file in config_file_list:

        try:
            # create a new csv file for each api call
            with open("single_task_api_call.csv", "w") as f:
                writer = csv.writer(f)
                writer.writerow(["api_cost"])
            
            env = WebArenaEnvironmentWrapper(config_file=config_file, 
                                            max_browser_rows=config.env.max_browser_rows, 
                                            max_steps=config.env.max_env_steps, 
                                            slow_mo=1, 
                                            observation_type="accessibility_tree", 
                                            current_viewport_only=True, 
                                            viewport_size={"width": 1920, "height": 1080}, 
                                            headless=config.env.headless)
            
            agent = agent_init()
            objective = env.get_objective()
            status = agent.act(objective=objective, env=env)
            env.close()
        
        except Exception as e:
            env.close()
            status = {'done': False, 'reward': 0.0, 'success': 0.0, 'num_actions': 0, 'action_limit_exceeded': False, "error": str(e)}

        if config.logging:
            with open(config_file, "r") as f:
                task_config = json.load(f)
            log_file = os.path.join(dstdir, f"{task_config['task_id']}.json")
            log_data = {
                "task": config_file,
                "id": task_config['task_id'],
                "model": config.agent.model_name,
                "type": config.agent.type,
                "trajectory": agent.get_trajectory(),
            }
            summary_file = os.path.join(dstdir, "summary.csv")

            ########## MZJ: cal api cost ##########
            # calculate the total api cost, and write into it
            with open("single_task_api_call.csv", "r") as f:
                lines = f.readlines()[1:]
                total_api_cost = sum([float(line) for line in lines])
                total_api_call_num = len(lines)
            with open("single_task_api_call.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([total_api_cost])
            # rename and move the csv file to the log directory
            os.rename("single_task_api_call.csv", os.path.join(dstdir, f"cost_{task_config['task_id']}.csv"))
            ########## MZJ: end ##########

            summary_data = {
                "task": config_file,
                "task_id": task_config['task_id'],
                "model": config.agent.model_name,
                "type": config.agent.type,
                "logfile": re.search(r"/([^/]+/[^/]+\.json)$", log_file).group(1),
                "api_cost": total_api_cost,
                "api_call_num": total_api_call_num,
            }
            summary_data.update(status)
            log_run(
                log_file=log_file,
                log_data=log_data,
                summary_file=summary_file,
                summary_data=summary_data,
            )
            
        # For reddit: Sleep for 21 minutes (720 seconds) 
        # time.sleep(1260)
    
if __name__ == "__main__":
    run()
