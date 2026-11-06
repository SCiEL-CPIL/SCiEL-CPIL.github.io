"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time


def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling.
    Combines multiple heuristics to pick the best ready operation to start next.
    
    Strategy:
    1. For each ready operation, compute a score based on:
       - Shortest processing time (SPT) on the chosen machine
       - Remaining work for the job (to prioritize jobs with more work)
       - Number of remaining operations (to break ties)
    2. Select the operation with the best composite score.
    
    The key insight is that in job-shop scheduling, we want to:
    - Start operations that can be completed quickly (SPT) to free up machines
    - Prioritize jobs with more remaining work to avoid long tail jobs
    - Consider machine load (though we don't have direct access to it here)
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Compute scores for each ready operation
    scores = np.zeros(n)
    
    for i, op in enumerate(ready_operations):
        # Get machine options and find the minimum duration (SPT metric)
        machine_options = op["machine_options"]
        durations = list(machine_options.values())
        min_duration = min(durations)
        
        # Remaining work for this job
        remaining_work = op["remaining_work"]
        
        # Number of remaining operations
        remaining_ops = op["remaining_ops"]
        
        # Composite score:
        # - Primary: shorter processing time is better (lower is better)
        # - Secondary: more remaining work is better (higher is better) 
        # - Tertiary: fewer remaining ops is better (lower is better)
        
        # Normalize to make them comparable
        # We want to minimize min_duration, maximize remaining_work, minimize remaining_ops
        # So score = -min_duration + remaining_work - remaining_ops
        # But we need to balance the scales
        
        # Use a weighted combination
        # SPT weight: 1.0 (most important)
        # Remaining work weight: 0.1 (secondary)
        # Remaining ops weight: -0.05 (tertiary)
        
        scores[i] = -1.0 * min_duration + 0.1 * remaining_work - 0.05 * remaining_ops
    
    # Select the operation with the highest score
    best_idx = int(np.argmax(scores))
    
    return best_idx
# EVOLVE-BLOCK-END
