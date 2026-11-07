"""
Bootstrap spawn: Direct approach using theoretical bin packing insights.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()

def heuristics(demands, capacity):
    """
    Returns an item-by-item matrix of attractiveness for placing item j right after item i.
    
    Strategy:
    Combined fill-ratio and complementarity heuristic:
    1. Fill ratio term: (pair_sums / capacity)^alpha rewards high utilization.
    2. Complementarity term: rewards pairs with different sizes, which tend to
       pack more efficiently by filling gaps left by same-sized items.
    
    The product of these terms creates a multi-dimensional attractiveness score.
    """
    n = len(demands)
    d = np.asarray(demands, dtype=np.float64)
    
    # Compute pairwise sums
    pair_sums = d[:, None] + d[None, :]
    
    # Determine which pairs fit in a bin
    fits = pair_sums <= capacity
    
    # Wasted space: capacity - pair_sum (for fitting pairs)
    # We want to minimize waste, so attractiveness should be high when waste is low
    waste = capacity - pair_sums
    
    # For fitting pairs, waste is in [0, capacity - 40] = [0, 110]
    # Use exponential decay: exp(-gamma * waste / capacity)
    # This gives strong preference to near-full bins
    gamma = 12.0
    fill_term = np.exp(-gamma * waste / capacity)
    
    # For non-fitting pairs, set to 0 (will be masked to 1e-9)
    fill_term = np.where(fits, fill_term, 0.0)
    
    # Add a small bonus for pairs where both items are "large" (hard to place)
    # This helps ensure large items get placed early
    min_demand = np.minimum(d[:, None], d[None, :])
    max_demand = np.maximum(d[:, None], d[None, :])
    
    # Bonus for pairs where the smaller item is still relatively large
    # Normalized: (min_demand - 20) / 79
    size_bonus = 1.0 + 0.3 * (min_demand - 20.0) / 79.0
    
    # Combine multiplicatively
    H_fit = fill_term * size_bonus
    
    # Mask out non-fitting pairs
    H = np.where(fits, H_fit, 1e-9)
    
    # Ensure no non-finite values
    H = np.where(np.isfinite(H), H, 1e-9)
    
    # The task says "Heuristic entries that are not positive become 1e-9"
    H = np.where(H > 0, H, 1e-9)

    return H

# EVOLVE-BLOCK-END
