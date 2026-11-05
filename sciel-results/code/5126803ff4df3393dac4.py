"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START_TIME = _time.time()
_TIME_LIMIT = 145.0  # safety margin before 150s kill


def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling.
    
    Uses a refined composite heuristic:
    - Primary: remaining work (critical path priority)
    - Secondary: prefer operations with shorter min duration to reduce blocking
    - Tertiary: prefer operations with fewer remaining ops (closer to completion)
    - Quaternary: slight preference for constrained operations (fewer machine options)
    
    Key insight: In job-shop scheduling, the critical path dominates makespan.
    We weight remaining work heavily but also consider that operations with
    fewer remaining steps should be prioritized to avoid idle machines.
    """
    if not ready_operations:
        return 0
    
    # Deadline guard
    if _time.time() - _START_TIME > _TIME_LIMIT:
        return 0
    
    n = len(ready_operations)
    
    # If only one operation, no choice
    if n == 1:
        return 0
    
    scores = np.empty(n)
    
    for i, op in enumerate(ready_operations):
        remaining_work = op["remaining_work"]
        machine_options = op["machine_options"]
        remaining_ops = op["remaining_ops"]
        
        # Find the minimum processing time across available machines
        if machine_options:
            min_duration = min(machine_options.values())
            num_options = len(machine_options)
        else:
            min_duration = 1
            num_options = 1
        
        # Composite score:
        # - High remaining work is good (priority to critical path)
        # - Shorter min duration is good (frees machine sooner)
        # - Fewer remaining ops is good (closer to completion, less risk)
        # - Fewer machine options is slightly better (constrained operations first)
        
        # Normalize remaining_ops: fewer remaining ops = higher score
        # remaining_ops ranges from 1 to ~10 for 10 jobs
        remaining_ops_factor = 1.0 / max(remaining_ops, 1)
        
        # Weighted composite:
        # remaining_work dominates (critical path)
        # min_duration has moderate negative weight
        # remaining_ops_factor has positive weight (completing soon)
        # num_options has small negative weight (constrained first)
        
        score = remaining_work * 1.0
        score -= min_duration * 0.15
        score += remaining_ops_factor * 0.3
        score -= (num_options - 1) * 0.02
        
        scores[i] = score
    
    # Return index of max score
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
