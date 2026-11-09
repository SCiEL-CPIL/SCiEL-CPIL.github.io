"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as time_module

def dispatch(ready_operations, time):
    """
    Dispatch rule for Flexible Job-Shop Scheduling.
    
    Heuristic: Maximize (Remaining Work / Min Duration).
    This prioritizes jobs with significant remaining work that can be started
    with a short processing step, balancing job completion progress with
    machine utilization efficiency.
    """
    if not ready_operations:
        return 0

    best_index = 0
    best_score = -1.0
    best_min_dur = float('inf')
    
    for i, op in enumerate(ready_operations):
        # Calculate the minimum duration among eligible machines
        # This assumes the simulator will pick the fastest machine or we
        # approximate the effective duration by the minimum.
        durations = list(op["machine_options"].values())
        if not durations:
            min_dur = 0
        else:
            min_dur = min(durations)
        
        rem_work = op["remaining_work"]
        
        # Score: Ratio of remaining work to the cost of the next step.
        # High rem_work (big job) and low min_dur (fast step) = High Score.
        if min_dur == 0:
            score = float('inf')
        else:
            score = rem_work / min_dur
            
        # Tie-breaking: If scores are close, prefer smaller min_dur (SPT)
        if score > best_score or (abs(score - best_score) < 1e-9 and min_dur < best_min_dur):
            best_score = score
            best_min_dur = min_dur
            best_index = i

    return int(best_index)
# EVOLVE-BLOCK-END
