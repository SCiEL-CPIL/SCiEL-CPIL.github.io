"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def dispatch(ready_operations, time):
    """
    Dispatch rule for open-shop scheduling.
    
    Strategy:
    1. Primary criterion: Most remaining work (MWL) - proven effective for makespan.
    2. Secondary tie-breaker: Prefer operations with fewer remaining ops (to finish jobs sooner).
    3. Tertiary tie-breaker: Prefer operations with shorter minimum duration on available machines.
    
    This composite heuristic balances load balancing with job completion priority.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    
    # Compute scores for each ready operation
    scores = []
    for op in ready_operations:
        work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        
        # Get durations from machine options
        durations = list(op["machine_options"].values())
        if durations:
            min_dur = min(durations)
            max_dur = max(durations)
        else:
            min_dur = 1.0
            max_dur = 1.0
        
        # Composite score:
        # - Primary: maximize remaining work (bigger jobs first)
        # - Secondary: minimize remaining_ops (finish jobs with fewer ops left)
        # - Tertiary: minimize min_dur (shorter operations first to unlock dependencies)
        
        # Normalize-ish combination
        score = work * 100.0 - remaining_ops * 10.0 - min_dur * 1.0
        
        scores.append(score)
    
    # Return index of operation with highest score
    best_idx = int(np.argmax(scores))
    return best_idx
# EVOLVE-BLOCK-END
