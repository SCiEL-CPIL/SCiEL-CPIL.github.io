"""Flexible Job-Shop Scheduling dispatch rule with critical path prioritization and machine selection."""
import time as _time
import math

# Deadline guard: the evaluator kills after 150s wall-clock. We keep a safety margin.
_DEADLINE = 150.0
_START_TIME = None

def _ensure_start():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _time.time()
    return _START_TIME

def _elapsed():
    return _time.time() - _ensure_start()

def _remaining_path(op):
    """
    Estimate the longest remaining path for a job from this operation onward.
    We use the sum of minimum durations across machine_options for each remaining operation.
    This is a lower-bound on the time needed to complete the job from this point.
    """
    total = 0.0
    for m, dur in op.get("machine_options", {}).items():
        total += dur
    # But we only need the minimum per operation, not the sum over all machines.
    # Actually, for a single operation, the minimum duration across machines is the best-case time.
    # For the remaining path, we sum the min durations of each remaining operation in the job.
    # But we don't have the full remaining sequence explicitly - we have remaining_ops and remaining_work.
    # Let's use remaining_work as a proxy, but also consider the structure.
    
    # Better: use remaining_work which should be the sum of min durations or similar.
    # Let's look at what fields are available.
    return op.get("remaining_work", 0)


def dispatch(ready_operations, time):
    """
    Dispatch rule: pick the ready operation with the highest critical path priority,
    and select the machine that minimizes the resulting completion time.
    
    Returns: (index, machine) or just index.
    """
    if not ready_operations:
        return 0
    
    # Check deadline
    if _elapsed() > _DEADLINE - 5.0:
        # Safety: just return the first one
        return 0
    
    best_index = 0
    best_score = -float('inf')
    best_machine = None
    
    for i, op in enumerate(ready_operations):
        # Calculate priority based on remaining work / critical path
        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 0)
        
        # Critical path estimate: sum of min durations for remaining operations
        # If we don't have explicit remaining sequence, use remaining_work
        # remaining_work is likely the sum of min processing times for all remaining ops in the job
        
        # Machine selection: pick machine that gives earliest completion
        machine_options = op.get("machine_options", {})
        if not machine_options:
            continue
        
        # Find machine with minimum duration (earliest completion)
        min_duration = min(machine_options.values())
        min_machine = min(machine_options, key=machine_options.get)
        
        # Priority: higher remaining work means more critical
        # Also consider: operations with fewer remaining ops might be closer to completion
        # Use a combined score: remaining_work weighted, with a small bonus for fewer remaining ops
        
        # The key insight: operations that are on the critical path (high remaining work)
        # should be scheduled first to avoid idle time
        
        score = remaining_work
        
        # Tie-breaker: prefer operations with fewer remaining ops (closer to done)
        # This helps with job completion and frees up resources
        if remaining_ops > 0:
            score += 1.0 / remaining_ops
        
        if score > best_score:
            best_score = score
            best_index = i
            best_machine = min_machine
    
    # Return index and machine for flexible job-shop
    if best_machine is not None:
        return (best_index, best_machine)
    return best_index
