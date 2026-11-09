"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
# EVOLVE-BLOCK-START
def dispatch(ready_operations, time):
    """
    Dispatch rule using Shortest Processing Time (SPT) on the minimum possible
    duration of the operation, with tie-breaking by Least Remaining Work.
    
    This approach prioritizes operations that can be completed quickly (min duration),
    which helps in freeing up job slots and potentially reducing makespan by
    completing jobs sooner. If durations are equal, it prefers jobs with less
    total remaining work.
    """
    if not ready_operations:
        return 0

    scores = []
    for op in ready_operations:
        # Get the shortest possible duration for this operation
        min_duration = min(op["machine_options"].values()) if op["machine_options"] else 0.0
        
        # Remaining work: total processing time left for this job
        rem_work = op["remaining_work"]
        
        # Score: Primary key is min_duration (SPT).
        # Secondary key is rem_work (LRWF) to break ties.
        # We scale min_duration heavily to ensure it dominates.
        # Assuming durations are > 0 and rem_work is comparable in scale,
        # we can use a large multiplier for duration.
        score = min_duration * 1000 + rem_work
        scores.append(score)
    
    # Return the index of the operation with the minimum score
    return int(np.argmin(scores))
# EVOLVE-BLOCK-END
