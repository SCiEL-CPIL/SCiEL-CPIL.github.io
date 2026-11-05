"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule based on Critical Ratio (CR) heuristic.
    
    For each ready operation, we estimate slack time as:
        slack = remaining_work - time (where remaining_work is the total remaining processing time for the job)
    
    The Critical Ratio is: CR = slack / remaining_work
    
    Lower CR means more urgent (less slack relative to work remaining).
    We pick the operation with the lowest CR (most urgent).
    
    If slack <= 0, the job is already "late" in terms of our proxy deadline, so it gets very high priority.
    """
    if not ready_operations:
        return 0
    
    scores = []
    for op in ready_operations:
        remaining_work = op["remaining_work"]
        
        # Estimate slack: how much time is left before the job's "deadline"
        # Using time as a proxy for elapsed time, slack = remaining_work - time
        # This means jobs with more remaining work relative to elapsed time have more slack
        slack = remaining_work - time
        
        # Critical Ratio = slack / remaining_work
        # If remaining_work is 0 or very small, handle edge case
        if remaining_work <= 0:
            # No remaining work, should not be in ready list, but handle gracefully
            cr = float('inf')
        else:
            cr = slack / remaining_work
        
        scores.append(cr)
    
    # Pick the operation with the lowest CR (most urgent)
    best_idx = int(np.argmin(scores))
    return best_idx
# EVOLVE-BLOCK-END
