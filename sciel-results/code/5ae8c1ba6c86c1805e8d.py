"""Rewritten dispatch rule for job-shop scheduling."""
# EVOLVE-BLOCK-START
import time as _time
import math

# Global deadline guard
_START_TIME = _time.time()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin before 150s limit


def _is_deadline_exceeded():
    return _time.time() > _DEADLINE


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Uses a multi-criteria heuristic:
    1. Minimum processing time on best available machine (to reduce idle time)
    2. Weighted Shortest Processing Time (WSPT) - prioritize jobs with more remaining work
    3. Fewer remaining operations (to complete jobs sooner)
    
    Returns the index of the chosen operation.
    """
    if not ready_operations:
        return 0

    if _is_deadline_exceeded():
        return 0

    n = len(ready_operations)
    if n == 1:
        return 0

    best_idx = 0
    best_score = float('-inf')

    for i, op in enumerate(ready_operations):
        options = op.get('machine_options', {})
        if not options:
            continue

        # Find the minimum duration across available machines
        min_dur = min(options.values())
        
        # Get remaining work and remaining ops
        remaining_work = op.get('remaining_work', 0)
        remaining_ops = op.get('remaining_ops', 0)

        # Composite score:
        # - Primary: minimize processing time (shorter is better)
        # - Secondary: prioritize jobs with more remaining work (WSPT-like)
        # - Tertiary: prioritize jobs with fewer remaining ops (to finish them)
        
        # Normalize components to [0, 1] range for fair combination
        # Shorter min_dur is better -> use inverse or negative
        # More remaining_work is better -> use positive
        # Fewer remaining_ops is better -> use negative
        
        # Use a weighted sum where we want to maximize the score
        # score = w1 * (1/min_dur) + w2 * remaining_work - w3 * remaining_ops
        
        # To avoid division by zero, add small epsilon
        eps = 1e-9
        score = (1.0 / (min_dur + eps)) * 100.0 + remaining_work * 0.5 - remaining_ops * 5.0

        if score > best_score:
            best_score = score
            best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
