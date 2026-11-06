"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule: Longest Remaining Processing Time (LRPT) with tie-breaking by 
    Shortest Processing Time (SPT).
    
    This rule prioritizes operations from jobs that have the most remaining work,
    which helps avoid delaying jobs that are close to completion but have long
    remaining sequences. This is often more effective in job-shop scheduling than
    pure SPT because it prevents "starvation" of jobs that still have significant
    work left.
    
    For each ready operation, we look at the job's remaining_work. We prioritize
    the operation with the maximum remaining_work. In case of ties, we pick the
    one with the shortest processing time (min duration across machines) to keep
    machines busy with quick tasks when possible.
    """
    if not ready_operations:
        return 0

    remaining_works = []
    min_durations = []
    
    for op in ready_operations:
        # Get the minimum duration across all machine options
        if op["machine_options"]:
            min_dur = min(op["machine_options"].values())
        else:
            min_dur = 0
        min_durations.append(min_dur)
        remaining_works.append(op["remaining_work"])

    remaining_works = np.array(remaining_works)
    min_durations = np.array(min_durations)

    # Primary key: Maximize remaining_work
    # Secondary key: Minimize min_duration
    
    max_rwork = np.max(remaining_works)
    # Mask of operations that have the maximum remaining work
    mask = remaining_works == max_rwork
    
    if np.sum(mask) == 1:
        return int(np.where(mask)[0][0])
    
    # If multiple, pick the one with min duration among the masked ones
    candidates = np.where(mask)[0]
    
    # Among candidates, find the one with min duration
    best_idx = candidates[np.argmin(min_durations[candidates])]
    
    return int(best_idx)

# EVOLVE-BLOCK-END
