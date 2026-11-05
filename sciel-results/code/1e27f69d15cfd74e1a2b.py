"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule based on a Critical Ratio (CR) heuristic.
    
    The heuristic prioritizes operations with the lowest ratio of estimated slack time 
    to remaining processing time.
    
    Slack is estimated as: remaining_work - current_time.
    Remaining processing time is the total remaining work for the job.
    
    Ratio = (remaining_work - time) / remaining_work
    
    Minimizing this ratio effectively prioritizes jobs with the least remaining work 
    (Shortest Remaining Work), which helps in reducing makespan by completing jobs 
    earlier and freeing up resources.
    
    Tie-breaker: Prefer operations with shorter minimum processing time.
    """
    if not ready_operations:
        return 0
    
    scores = []
    min_durations = []
    
    for op in ready_operations:
        remaining_work = op["remaining_work"]
        
        # Avoid division by zero, though remaining_work should be > 0 for ready ops
        if remaining_work <= 0:
            ratio = float('inf')
        else:
            # Slack = Remaining Work - Time
            # We want the "lowest" ratio.
            # Minimizing (RW - T)/RW is equivalent to minimizing RW.
            ratio = (remaining_work - time) / remaining_work
        
        scores.append(ratio)
        
        # Get min duration for tie-breaking
        machine_options = op["machine_options"]
        if machine_options:
            min_duration = min(machine_options.values())
        else:
            min_duration = 1.0
        min_durations.append(min_duration)
    
    # Find the index with the lowest ratio
    # If there are ties, we want the one with the lowest min_duration.
    # We can combine them into a single score for sorting.
    # Primary: ratio (lower is better)
    # Secondary: min_duration (lower is better)
    
    # Create a list of tuples (ratio, min_duration, index)
    candidates = list(zip(scores, min_durations, range(len(ready_operations))))
    
    # Sort by ratio, then by min_duration
    candidates.sort(key=lambda x: (x[0], x[1]))
    
    best_idx = candidates[0][2]
    
    return best_idx
# EVOLVE-BLOCK-END
