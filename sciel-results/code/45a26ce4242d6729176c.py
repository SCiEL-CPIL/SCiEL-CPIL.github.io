"""
Bootstrap spawn: Direct approach using theoretical bin packing insights.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0  # Safety margin before 300s limit

def heuristics(demands, capacity):
    """
    Returns an item-by-item matrix of attractiveness for placing item j right after item i.
    
    Strategy:
    Combines two signals:
    1. Fill-ratio power: strongly rewards pairs whose combined size is close to capacity.
    2. Complementarity bonus: rewards pairs where items have different sizes, since
       heterogeneous pairs leave gaps that are easier to fill with other items.
    
    The complementarity term uses the normalized size difference between the two items.
    This prevents the heuristic from being overly narrow (which can trap ACO in local optima)
    while still strongly favoring high-utilization pairs.
    """
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE - 5.0:
        # Emergency fallback: return a simple valid matrix
        n = len(demands)
        return np.full((n, n), 1e-9, dtype=np.float64)
    
    n = len(demands)
    d = np.asarray(demands, dtype=np.float64)
    
    # Compute pairwise sums
    pair_sums = d[:, None] + d[None, :]
    
    # Determine which pairs fit in a bin
    fits = pair_sums <= capacity
    
    # Fill ratio for fitting pairs
    ratios = pair_sums / capacity
    
    # Primary signal: power of fill ratio
    # alpha=12 is strong but not as extreme as 20, allowing more exploration
    alpha = 12.0
    fill_term = np.power(ratios, alpha)
    
    # Complementarity term: reward size difference
    # When items have very different sizes, the gap left is more "fillable"
    # Normalize the difference to [0, 1]
    size_diff = np.abs(d[:, None] - d[None, :])
    # Max possible diff is 99 - 20 = 79
    max_diff = 79.0
    norm_diff = size_diff / max_diff
    
    # Complementarity bonus: a gentle boost for pairs with different sizes
    # This is a smooth term that adds up to ~30% extra attractiveness
    comp_weight = 0.3
    comp_term = 1.0 + comp_weight * norm_diff
    
    # Combined heuristic for fitting pairs
    H_fit = fill_term * comp_term
    
    # For non-fitting pairs, set to minimum
    H = np.where(fits, H_fit, 1e-9)
    
    # Ensure no non-finite values
    H = np.where(np.isfinite(H), H, 1e-9)
    
    # Ensure positive values
    H = np.where(H > 0, H, 1e-9)
    
    return H

# EVOLVE-BLOCK-END
