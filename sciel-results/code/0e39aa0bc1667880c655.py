"""Optimized dispatch rule for open-shop scheduling."""
# EVOLVE-BLOCK-START
import time
import random
import sys
import math
from collections import defaultdict

# Global state for the simulator (per evaluation run)
_sim_state = {}
_DEADLINE = 115.0  # Safety margin before 120s limit
_start_time = None

def _reset_sim_state():
    global _sim_state, _start_time
    _sim_state = {}
    _start_time = time.time()

def _check_time():
    if _start_time is None:
        return False
    return (time.time() - _start_time) > _DEADLINE

def dispatch(ready_operations, time):
    """
    Dispatch rule that selects which ready operation to start next.
    
    Uses a weighted scoring approach combining:
    - Most Work Remaining (MWR) for job prioritization
    - Most Operations Remaining (MOR) to avoid job completion delays
    - SPT (Shortest Processing Time) preference for machine selection
    """
    # Initialize global state on first call
    if _start_time is None:
        _reset_sim_state()
    
    if not ready_operations:
        return 0
    
    # Time guard: if we're running out of time, use a fast simple rule
    if _check_time():
        return max(range(len(ready_operations)), 
                  key=lambda i: ready_operations[i]["remaining_work"])
    
    # Parameters tuned for open-shop scheduling
    w_work = 1.0
    w_ops = 0.2
    w_spt = 0.15
    
    scores = []
    for i, op in enumerate(ready_operations):
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        
        machine_options = op["machine_options"]
        if machine_options:
            durations = list(machine_options.values())
            min_duration = min(durations)
            # SPT bonus: higher score for shorter min duration
            spt_bonus = 1.0 / (min_duration + 1e-9)
        else:
            spt_bonus = 0.0
        
        # Score: prioritize high remaining work, more remaining ops, shorter min processing
        score = (w_work * remaining_work + 
                 w_ops * remaining_ops * 5 + 
                 w_spt * spt_bonus)
        
        scores.append(score)
    
    best_idx = max(range(len(ready_operations)), key=lambda i: scores[i])
    return int(best_idx)

# EVOLVE-BLOCK-END
