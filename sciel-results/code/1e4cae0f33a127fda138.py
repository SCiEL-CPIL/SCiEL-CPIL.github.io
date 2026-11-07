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
    
    # Compute pairwise sums and absolute differences
    pair_sums = d[:, None] + d[None, :]
    pair_diffs = np.abs(d[:, None] - d[None, :])
    
    # Determine which pairs fit in a bin
    fits = pair_sums <= capacity
    
    # Fill ratio: (pair_sums / capacity)^alpha
    alpha = 15.0
    ratios = pair_sums / capacity
    fill_term = np.power(ratios, alpha)
    
    # Complementarity term: 1 + beta * (normalized size difference)
    # Normalize by max possible difference (99 - 20 = 79)
    max_diff = 79.0  # Since demands are in [20, 99]
    beta = 0.8
    complementarity = 1.0 + beta * (pair_diffs / max_diff)
    
    # Combine: multiplicative combination
    H_fit = fill_term * complementarity
    
    # Mask out non-fitting pairs
    H = np.where(fits, H_fit, 1e-9)
    
    # Ensure no non-finite values
    H = np.where(np.isfinite(H), H, 1e-9)
    
    # The task says "Heuristic entries that are not positive become 1e-9"
    H = np.where(H > 0, H, 1e-9)

    return H

# EVOLVE-BLOCK-END
