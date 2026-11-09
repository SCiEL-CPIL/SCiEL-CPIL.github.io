"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Track elapsed time to enforce deadline
_start_time = None

def _elapsed_ok(deadline_margin=5.0):
    """Check if we're still within the time budget."""
    global _start_time
    if _start_time is None:
        _start_time = _time.time()
    return (_time.time() - _start_time) < (150.0 - deadline_margin)


def dispatch(ready_operations, time):
    """
    Dispatch rule for open-shop scheduling.
    
    Strategy: Prioritize operations that have:
    1. High remaining work (to finish jobs quickly)
    2. Can be assigned to a less-loaded machine (to balance load)
    3. Have fewer machine options (to reduce flexibility loss)
    
    We use a weighted score that combines these factors.
    """
    if not ready_operations:
        return 0
    
    # Check deadline - if close to timeout, use fast heuristic
    if not _elapsed_ok(deadline_margin=10.0):
        # Fast fallback: most remaining work
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    n_ops = len(ready_operations)
    scores = np.zeros(n_ops)
    
    # Get all machines and their current loads (estimated from time)
    # In open-shop, we don't directly know machine loads, but we can estimate
    # by considering which machines have been busy
    
    for i, op in enumerate(ready_operations):
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        machine_options = op["machine_options"]
        
        # Factor 1: Remaining work (higher is better)
        work_score = remaining_work
        
        # Factor 2: Number of remaining operations (jobs with fewer remaining ops should be prioritized)
        ops_score = 1.0 / max(remaining_ops, 1)
        
        # Factor 3: Flexibility - operations with fewer machine options should be prioritized
        # (to avoid getting stuck later)
        num_options = len(machine_options)
        flex_score = 1.0 / max(num_options, 1)
        
        # Factor 4: Minimum duration among options (shorter processing times first)
        min_duration = min(machine_options.values()) if machine_options else 0
        duration_score = 1.0 / max(min_duration, 0.1)
        
        # Combine with weights
        # Weight: work=0.4, ops=0.3, flexibility=0.15, duration=0.15
        scores[i] = (0.4 * work_score + 
                     0.3 * ops_score + 
                     0.15 * flex_score + 
                     0.15 * duration_score)
    
    # If scores are all zero or invalid, fallback
    if not np.any(scores > 0):
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
