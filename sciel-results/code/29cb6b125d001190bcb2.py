"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(demands, capacity):
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    n = len(demands)
    pair = demands[:, None] + demands[None, :]
    waste = capacity - pair
    
    # Base: utilization squared (rewards high fill)
    util = pair / capacity
    base = util * util
    
    # Efficiency bonus: reward low waste
    waste_ratio = waste / capacity
    efficiency_bonus = 1.0 + 0.5 * (1.0 - waste_ratio)
    
    # Combined heuristic: base * efficiency_bonus
    heuristic_vals = base * efficiency_bonus
    
    # Only keep valid pairs (sum <= capacity)
    result = np.where(pair <= capacity, heuristic_vals, 1e-9)
    
    # Ensure all values are positive and finite
    result = np.where(result > 0, result, 1e-9)
    result = np.where(np.isfinite(result), result, 1e-9)
    
    # Check time budget
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        # Return best result found so far (which is already computed)
        pass
    
    return result
# EVOLVE-BLOCK-END
