"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule: Weighted Shortest Processing Time (WSPT) with Critical Path weighting.
    
    This approach uses a ratio-based heuristic distinct from linear combinations.
    It prioritizes operations that have a high ratio of remaining work to processing time.
    This implicitly balances the critical path length (remaining work) with the
    efficiency of starting short operations (min duration) to reduce machine blocking.
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Precompute scores for each ready operation
    scores = []
    for op in ready_operations:
        job = op["job"]
        machine_options = op["machine_options"]
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        
        # Minimum duration to complete this operation (best case)
        # Use 1.0 as a floor to avoid division by zero and to handle 0 duration
        min_duration = min(machine_options.values()) if machine_options else 0
        effective_duration = max(min_duration, 1.0)
        
        # WSPT Score: Prioritize high remaining work relative to processing time
        # Higher score is better
        wspt_score = remaining_work / effective_duration
        
        # Add a small bonus for jobs closer to completion to reduce active job count
        # This acts as a tie-breaker favoring finishing jobs
        completion_bonus = 1.0 / max(remaining_ops, 1)
        
        # Combine: Dominant WSPT score, minor completion bonus
        score = wspt_score + completion_bonus
        
        scores.append(score)
    
    # Select the operation with the highest score
    best_idx = int(np.argmax(scores))
    
    return best_idx
# EVOLVE-BLOCK-END
