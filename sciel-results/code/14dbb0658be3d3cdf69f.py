"""Optimized dispatch rule for open-shop scheduling."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

# Global state for tracking simulation progress
_sim_state = {}
_DEADLINE = 115.0
_start_time = None

def _reset_sim_state():
    global _sim_state, _start_time
    _sim_state = {
        'call_count': 0,
    }
    _start_time = time.time()

def _check_time():
    if _start_time is None:
        return False
    return (time.time() - _start_time) > _DEADLINE

def dispatch(ready_operations, time):
    """
    Dispatch rule for open-shop scheduling.
    
    Strategy:
    1. Job Selection: Weighted combination of Most Work Remaining (MWR), 
       Most Operations Remaining (MOR), and Shortest Processing Time (SPT) potential.
       - MWR: Prioritizes jobs with the most remaining work to reduce makespan.
       - MOR: Prioritizes jobs with more remaining operations to keep them flowing.
       - SPT: Prioritizes operations that can be completed quickly (low min duration) 
         to reduce blocking and idle time.
    2. Machine Selection: Shortest Processing Time (SPT) for the selected job.
       - This is a robust heuristic for open-shop problems as it minimizes the time 
         a job occupies a specific machine, allowing other operations to be scheduled.
    
    Returns: (index, machine) tuple
    """
    # Initialize on first call
    if _start_time is None:
        _reset_sim_state()
    
    if not ready_operations:
        return (0, None)
    
    # Time guard: if we're running out of time, use a fast simple rule
    if _check_time():
        # Fallback: MWR
        best_idx = max(range(len(ready_operations)), 
                      key=lambda i: ready_operations[i]["remaining_work"])
        op = ready_operations[best_idx]
        machine_options = op.get("machine_options", {})
        if machine_options:
            # Fallback machine: SPT
            best_machine = min(machine_options.keys(), key=lambda m: machine_options[m])
        else:
            best_machine = None
        return (best_idx, best_machine)
    
    state = _sim_state
    state['call_count'] += 1
    
    scores = []
    best_machines = []
    
    # Weights for the scoring function
    # Tuned for open-shop scheduling:
    # MWR is the dominant factor.
    # SPT potential helps in clearing quick operations.
    # MOR ensures jobs don't get stuck.
    w_work = 1.0
    w_ops = 0.5
    w_spt = 0.5
    w_flex = 0.1
    
    # Pre-compute max values for normalization if needed, 
    # but raw values often work well if scales are similar.
    # To be safe, we can normalize or just use raw.
    # Let's use raw values as they are comparable in magnitude for 10 jobs/5 machines.
    
    for i, op in enumerate(ready_operations):
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        
        machine_options = op.get("machine_options", {})
        if not machine_options:
            scores.append(0)
            best_machines.append(None)
            continue
        
        durations = list(machine_options.values())
        min_duration = min(durations)
        num_machines = len(machine_options)
        
        # Component 1: Most Work Remaining (MWR)
        # Higher is better
        work_score = remaining_work
        
        # Component 2: Most Operations Remaining (MOR)
        # Higher is better
        ops_score = remaining_ops
        
        # Component 3: SPT Potential
        # Shorter min duration is better. 
        # Use 1/duration to make it a "score" where higher is better.
        # Add epsilon to avoid division by zero.
        spt_score = 1.0 / (min_duration + 1e-9)
        
        # Component 4: Flexibility
        # More machine options is slightly better as it provides more scheduling freedom.
        flex_score = num_machines
        
        # Combined Score
        score = (w_work * work_score + 
                 w_ops * ops_score + 
                 w_spt * spt_score + 
                 w_flex * flex_score)
        
        # Determine best machine for this operation using SPT
        # If there's a tie in duration, pick the first one (arbitrary but consistent)
        best_machine = min(machine_options.keys(), key=lambda m: machine_options[m])
        
        scores.append(score)
        best_machines.append(best_machine)
    
    # Select the operation with the highest score
    best_idx = max(range(len(ready_operations)), key=lambda i: scores[i])
    return (best_idx, best_machines[best_idx])

# EVOLVE-BLOCK-END
