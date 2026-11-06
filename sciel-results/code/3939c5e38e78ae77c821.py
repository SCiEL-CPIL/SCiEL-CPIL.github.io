"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule: Shortest Processing Time (SPT) with tie-breaking by Remaining Work.
    
    For each ready operation, we determine the minimum possible processing time
    (assuming the simulator assigns the best available machine or the problem 
    effectively has a single best machine). We prioritize the operation with the 
    shortest processing time to minimize machine occupancy. In case of ties, 
    we prioritize the job with the most remaining work to avoid stranding long jobs.
    """
    if not ready_operations:
        return 0

    min_durations = []
    remaining_works = []
    
    for op in ready_operations:
        # Get the minimum duration across all machine options
        # If machine_options is empty (shouldn't happen), default to 0
        if op["machine_options"]:
            min_dur = min(op["machine_options"].values())
        else:
            min_dur = 0
        min_durations.append(min_dur)
        remaining_works.append(op["remaining_work"])

    min_durations = np.array(min_durations)
    remaining_works = np.array(remaining_works)

    # We want the smallest min_duration. 
    # If we use argmin, it returns the first occurrence. 
    # To handle tie-breaking by remaining_work (larger is better), we can 
    # create a composite score.
    
    # Strategy: 
    # 1. Find the global minimum duration.
    # 2. Among all ops with that duration, pick the one with max remaining_work.
    
    min_val = np.min(min_durations)
    # Mask of operations that have the minimum duration
    mask = min_durations == min_val
    
    if np.sum(mask) == 1:
        return int(np.where(mask)[0][0])
    
    # If multiple, pick the one with max remaining_work among the masked ones
    # We can use a trick: assign a very large penalty to non-min duration ops, 
    # or just filter.
    
    # Efficient way: 
    # Score = min_duration * (1 + max_remaining_work) - remaining_work
    # This is tricky to balance.
    
    # Simpler: 
    # 1. Get indices of min duration
    candidates = np.where(mask)[0]
    
    # Among candidates, find the one with max remaining_work
    best_idx = candidates[np.argmax(remaining_works[candidates])]
    
    return int(best_idx)

# EVOLVE-BLOCK-END
