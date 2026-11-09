"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as time_module

# Global start time for deadline guarding
_START_TIME = time_module.time()
_DEADLINE_LIMIT = 145.0  # 5 seconds margin before 150s kill

def dispatch(ready_operations, time):
    # Deadline Guard
    if time_module.time() - _START_TIME > _DEADLINE_LIMIT:
        # Fallback to seed behavior if we are running out of time
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    best_score = -1.0
    best_idx = 0
    
    for i, op in enumerate(ready_operations):
        # Find the minimum duration among eligible machines
        # This serves as a proxy for the "cost" or "load" of this operation
        min_dur = float('inf')
        for m, d in op['machine_options'].items():
            if d < min_dur:
                min_dur = d
        
        # Handle edge case where duration might be 0
        if min_dur <= 0:
            min_dur = 1e-6

        # Heuristic: Weighted Shortest Processing Time (WSPT) variant
        # Prioritize jobs with high remaining work (urgency) 
        # divided by the minimum possible processing time (efficiency).
        # This favors operations that are part of large jobs but can be completed quickly.
        score = op['remaining_work'] / min_dur
        
        if score > best_score:
            best_score = score
            best_idx = i
            
    return int(best_idx)
# EVOLVE-BLOCK-END
