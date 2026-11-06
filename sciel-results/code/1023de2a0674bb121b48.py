"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

_DEADLINE = 140.0  # Safety margin before 150s kill
_START_TIME = _time.monotonic()


def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling.
    Uses a weighted combination of:
    - Shortest Processing Time (SPT): prioritize short operations to free machines quickly
    - Remaining Work: prioritize jobs with more remaining work to avoid long-tail effects
    - Remaining Operations: break ties by fewer remaining steps
    
    This balances immediate machine utilization against overall job completion.
    """
    if not ready_operations:
        return 0
    
    # Deadline guard
    if _time.monotonic() - _START_TIME > _DEADLINE:
        return 0
    
    n = len(ready_operations)
    if n == 1:
        return 0
    
    best_score = -float('inf')
    best_idx = 0
    
    for i, op in enumerate(ready_operations):
        # Get machine options and find the minimum duration (SPT metric)
        machine_options = op["machine_options"]
        durations = list(machine_options.values())
        min_duration = min(durations) if durations else 1
        
        # Remaining work for this job
        remaining_work = op["remaining_work"]
        
        # Number of remaining operations
        remaining_ops = op["remaining_ops"]
        
        # SPT component: shorter operations get higher priority
        # Use inverse to make shorter = higher score
        if min_duration > 0:
            spt_score = 1.0 / min_duration
        else:
            spt_score = 1000.0
        
        # Remaining work component: jobs with more work get priority
        # Normalize by a factor to balance with SPT
        remaining_work_score = remaining_work * 0.1
        
        # Remaining ops component: fewer remaining ops get slight preference
        remaining_ops_score = -0.02 * remaining_ops
        
        # Composite score: SPT + remaining work + remaining ops
        score = spt_score + remaining_work_score + remaining_ops_score
        
        if score > best_score:
            best_score = score
            best_idx = i
    
    return best_idx
# EVOLVE-BLOCK-END
