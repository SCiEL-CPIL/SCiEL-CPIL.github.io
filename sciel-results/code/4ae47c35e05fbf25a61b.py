"""Rewritten dispatch rule for job-shop scheduling."""
# EVOLVE-BLOCK-START
import time as _time
import math

# Global deadline guard
_START_TIME = _time.time()
_DEADLINE = _START_TIME + 135.0  # 15s safety margin before 150s limit


def _is_deadline_exceeded():
    return _time.time() > _DEADLINE


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in a job-shop simulation.
    
    Uses a multi-criteria heuristic combining:
    1. WSPT (Weighted Shortest Processing Time) ratio: remaining_work / min_duration
       - Prioritizes operations that are "efficient" in terms of work per unit time
    2. Critical path / urgency: operations from jobs with fewer remaining operations
       - Completing jobs sooner reduces makespan
    3. Minimum processing time on best machine
       - Reduces machine idle time
    
    The heuristic scores each ready operation and picks the one with the best (lowest) score.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Check deadline early for very large problem instances
    if _is_deadline_exceeded():
        return 0

    best_idx = 0
    best_score = float('inf')

    for i, op in enumerate(ready_operations):
        # Minimum processing time across available machines
        machine_options = op["machine_options"]
        min_duration = min(machine_options.values())
        
        # Remaining work (total processing time for all remaining ops in this job)
        remaining_work = op["remaining_work"]
        
        # Number of remaining operations
        remaining_ops = op["remaining_ops"]
        
        # WSPT-like ratio: higher remaining work relative to current op duration
        # is less urgent; we want to prioritize ops where the current step is a
        # small fraction of remaining work (i.e., we're close to finishing the job)
        # Actually, for makespan, we want to finish jobs. So prioritize:
        # - Jobs with fewer remaining ops (closer to completion)
        # - Shorter current processing time (frees machine sooner)
        # - But also consider total remaining work to balance
        
        # Score components:
        # 1. Urgency: fewer remaining ops is better (lower score)
        urgency_score = remaining_ops
        
        # 2. Efficiency: shorter min duration is better (lower score)
        efficiency_score = min_duration
        
        # 3. WSPT: remaining_work / (min_duration + epsilon) - higher means
        #    more work left per unit time, so less urgent in some sense,
        #    but we want to balance. Actually, for makespan reduction,
        #    we want to complete jobs, so let's use:
        #    - Prioritize shorter remaining_ops
        #    - Then shorter min_duration
        #    - Then lower remaining_work (closer to done)
        
        # Composite score: weighted sum
        # Normalize-ish by using a combination
        # We want to minimize makespan, so:
        # - Complete jobs with few remaining ops first
        # - Among those, pick the one with shortest processing time
        # - Also consider that jobs with less total remaining work are closer to done
        
        # Score = urgency * weight1 + efficiency * weight2 + work_remaining * weight3
        # But we need to balance scales. Let's use:
        score = urgency_score * 10.0 + efficiency_score * 1.0 + remaining_work * 0.1
        
        # Alternative: use a ratio-based approach
        # WSPT: remaining_work / (min_duration + 1) - lower is better (less work per time unit)
        wspt_score = remaining_work / (min_duration + 1.0)
        
        # Combined: prioritize by urgency first, then WSPT
        combined_score = urgency_score * 5.0 + wspt_score * 1.0
        
        if combined_score < best_score:
            best_score = combined_score
            best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
