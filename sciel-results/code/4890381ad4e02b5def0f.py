"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule using a hybrid of Shortest Remaining Processing Time (SRPT)
    and Shortest Job First (SJF) based on remaining operation count.
    
    We prioritize operations that are short in duration (to free machines quickly)
    and belong to jobs with fewer remaining operations (to complete jobs faster).
    """
    if not ready_operations:
        return 0

    scores = []
    for op in ready_operations:
        # Get the shortest possible duration for this operation
        # In case machine_options has multiple entries, we assume the simulator
        # will pick the best one, but for scoring, we use the min as a proxy
        # for how quickly this op can finish.
        min_duration = min(op["machine_options"].values()) if op["machine_options"] else 0.0
        
        # Remaining ops count: how many more ops this job has after this one?
        # The dict key is 'remaining_ops'. If it includes the current one, 
        # a job with 1 op left is more urgent than 5.
        rem_ops = op["remaining_ops"]
        
        # Score: We want to MINIMIZE this.
        # Weight: Duration is in time units, rem_ops is a count.
        # We scale rem_ops down so that duration dominates if durations are large,
        # but if durations are similar, rem_ops breaks the tie.
        score = min_duration + 0.1 * rem_ops
        scores.append(score)
    
    # Return the index of the operation with the minimum score
    return int(np.argmin(scores))
# EVOLVE-BLOCK-END
