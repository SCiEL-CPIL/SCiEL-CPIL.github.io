"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Deadline guard: we must finish within 120s total, but dispatch is called many times.
# We keep a global start time and ensure we don't exceed a safe budget.
# However, since dispatch is called per-event and the total simulation time is bounded,
# we primarily need to ensure each dispatch call is fast.
# The 150s kill limit is for the entire evaluation. We'll add a lightweight guard.

_dispatch_start_time = None

def _ensure_init():
    global _dispatch_start_time
    if _dispatch_start_time is None:
        _dispatch_start_time = _time.time()

def _time_left():
    if _dispatch_start_time is None:
        return 115.0  # plenty of time initially
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
    # 1. Most Work Remaining (MWR) - primary
    # 2. Shortest Processing Time on best machine - to reduce makespan
    # 3. Consider machine load balance
    # 4. Consider remaining operations in the job (critical path)
    
    scores = []
    for op in ready_operations:
        work = op["remaining_work"]
        durations = list(op["machine_options"].values())
        if not durations:
            scores.append(0.0)
            continue
        
        min_dur = min(durations)
        max_dur = max(durations)
        avg_dur = sum(durations) / len(durations)
        num_machines = len(durations)
        remaining_ops = op.get("remaining_ops", 0)
        
        # Primary: Most Work Remaining
        score = work
        
        # Secondary: Prefer operations that can be done quickly (short min duration)
        # This helps unlock next operations sooner
        score += 0.15 * (1.0 / (min_dur + 1e-9))
        
        # Tertiary: Prefer operations with fewer remaining ops (closer to completion)
        # Completing jobs early can help with makespan
        if remaining_ops > 0:
            score += 0.1 * (1.0 / remaining_ops)
        
        # Quaternary: Slight penalty for operations with very long max duration
        # to avoid blocking a machine for too long
        score -= 0.05 * max_dur
        
        # Quinternary: Prefer operations with more machine options (flexibility)
        score += 0.02 * num_machines
        
        scores.append(score)
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
