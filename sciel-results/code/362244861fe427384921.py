"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

# Deadline guard: keep a hard wall-clock cap well under the 120s evaluator limit.
_DEADLINE_SECONDS = 110.0
_START_TIME = _time.monotonic()


def _elapsed():
    return _time.monotonic() - _START_TIME


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Strategy: Composite scoring that balances:
      1. Shortest Processing Time (SPT) on the chosen machine - reduces idle time
      2. Most remaining work (urgency) - avoids starving long jobs
      3. Fewest remaining operations - pushes jobs toward completion
      4. Earliest finish time on the selected machine - local machine efficiency
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # If only one option, return it immediately
    if n == 1:
        return 0
    
    # Check deadline
    if _elapsed() > _DEADLINE_SECONDS:
        return 0
    
    # Precompute scores for each ready operation
    scores = [0.0] * n
    
    # Find max remaining_work for normalization
    max_remaining_work = max(op["remaining_work"] for op in ready_operations)
    if max_remaining_work <= 0:
        max_remaining_work = 1.0
    
    # Find max remaining_ops for normalization
    max_remaining_ops = max(op["remaining_ops"] for op in ready_operations)
    if max_remaining_ops <= 0:
        max_remaining_ops = 1.0
    
    # Find max duration across all machine options for normalization
    max_duration = 0.0
    for op in ready_operations:
        for dur in op["machine_options"].values():
            if dur > max_duration:
                max_duration = dur
    if max_duration <= 0:
        max_duration = 1.0
    
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        
        # Choose the machine that gives the earliest finish time (min duration)
        # This is the machine we'd actually use
        best_machine = None
        best_duration = float('inf')
        for m, d in machine_options.items():
            if d < best_duration:
                best_duration = d
                best_machine = m
        
        if best_duration == float('inf'):
            best_duration = 1.0
        
        # Component 1: SPT - shorter operations get higher priority (1/duration)
        spt_score = 1.0 / (best_duration + 1e-9)
        
        # Component 2: Urgency - more remaining work gets higher priority
        urgency_score = op["remaining_work"] / max_remaining_work
        
        # Component 3: Fewer remaining ops - closer to completion
        # Invert: fewer remaining ops = higher score
        completion_score = 1.0 - (op["remaining_ops"] / max_remaining_ops)
        
        # Component 4: Earliest finish time consideration
        # Normalize the finish time (time + duration)
        finish_time = time + best_duration
        # Use a soft penalty for later finish times
        finish_score = 1.0 / (finish_time + 1.0)
        
        # Composite score with weights tuned for job-shop scheduling
        # SPT is strong for local efficiency, urgency prevents starvation
        w_spt = 0.35
        w_urgency = 0.30
        w_completion = 0.20
        w_finish = 0.15
        
        score = (w_spt * spt_score +
                 w_urgency * urgency_score +
                 w_completion * completion_score +
                 w_finish * finish_score)
        
        scores[i] = score
    
    # Return index of highest score
    best_idx = 0
    best_score = scores[0]
    for i in range(1, n):
        if scores[i] > best_score:
            best_score = scores[i]
            best_idx = i
    
    return int(best_idx)
# EVOLVE-BLOCK-END
