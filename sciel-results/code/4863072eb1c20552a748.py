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
    
    # First pass: collect all operations and their machine options
    # We'll use a smooth surrogate approach with log-sum-exp for machine load balancing
    scores = []
    for op in ready_operations:
        work = op["remaining_work"]
        machine_options = op["machine_options"]
        if not machine_options:
            scores.append(0.0)
            continue
        
        durations = list(machine_options.values())
        min_dur = min(durations)
        max_dur = max(durations)
        avg_dur = sum(durations) / len(durations)
        num_machines = len(durations)
        remaining_ops = op.get("remaining_ops", 0)
        
        # Primary: Most Work Remaining (MWR)
        # This is the dominant factor in open-shop scheduling
        score = work
        
        # Secondary: Prefer operations with shorter processing times
        # to keep machines busy and reduce idle time
        # Use min duration as a proxy for "quick to complete"
        score += 0.1 * (1.0 / (min_dur + 1e-9))
        
        # Tertiary: Prefer operations that are closer to completing their job
        # Completing a job frees it from consideration
        if remaining_ops > 0:
            score += 0.05 * (1.0 / remaining_ops)
        
        # Quaternary: Slight penalty for very long operations
        # to avoid blocking a machine for too long
        score -= 0.02 * max_dur
        
        # Quinary: Prefer operations with more machine options (flexibility)
        score += 0.01 * num_machines
        
        # Machine load balancing: Use soft-min to estimate the "effective" 
        # machine load if this operation were scheduled now.
        # We want to avoid scheduling operations that would make one machine 
        # significantly more loaded than others.
        # Use log-sum-exp as a smooth approximation to min to find the 
        # least-loaded machine option.
        # Higher score for operations that can be assigned to less-loaded machines.
        # Since we don't have direct machine load info, use the duration spread
        # as a proxy: operations with more balanced duration options are preferred.
        
        # Compute a smooth measure of machine option balance
        # Lower variance in durations means more flexibility in machine assignment
        if num_machines > 1:
            variance = sum((d - avg_dur) ** 2 for d in durations) / num_machines
            # Prefer operations with lower variance (more balanced options)
            score -= 0.005 * variance
        else:
            score -= 0.005 * 0.0
        
        # Additional: Use the ratio of min to max duration as a flexibility indicator
        # Higher ratio means more uniform options
        if max_dur > 0:
            flexibility_ratio = min_dur / max_dur
            score += 0.03 * flexibility_ratio
        
        scores.append(score)
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
