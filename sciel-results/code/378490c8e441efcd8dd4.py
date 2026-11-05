"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_dispatch_start_time = None

def _ensure_init():
    global _dispatch_start_time
    if _dispatch_start_time is None:
        _dispatch_start_time = _time.time()

def _time_left():
    if _dispatch_start_time is None:
        return 115.0
    return 115.0 - (_time.time() - _dispatch_start_time)

def dispatch(ready_operations, time):
    _ensure_init()
    
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # If we're running low on time, use the fastest heuristic
    if _time_left() < 5.0:
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    # Compute scores for each ready operation
    # Strategy: Combine multiple heuristics for open-shop scheduling
    # Key insight: In open-shop, we want to minimize makespan.
    # The critical path is determined by the longest job (sum of all ops) and 
    # the busiest machine (sum of all ops on that machine).
    # A good heuristic is to prioritize operations that are on the critical path.
    
    scores = []
    for op in ready_operations:
        work = op["remaining_work"]
        durations = list(op["machine_options"].values())
        if not durations:
            scores.append(0.0)
            continue
        
        min_dur = min(durations)
        max_dur = max(durations)
        remaining_ops = op.get("remaining_ops", 0)
        
        # Primary: Most Work Remaining (MWR)
        score = work
        
        # Secondary: Stronger preference for shorter current operation (SPT-like)
        # Scaled to have more influence than before
        score += 2.0 * (1.0 / (min_dur + 1.0))
        
        # Tertiary: Preference for jobs closer to completion
        if remaining_ops > 0:
            score += 0.5 * (1.0 / (remaining_ops + 1.0))
            
        # Quaternary: Small penalty for very long operations on the worst case
        # Reduced penalty compared to previous version
        score -= 0.01 * max_dur
        
        scores.append(score)
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
