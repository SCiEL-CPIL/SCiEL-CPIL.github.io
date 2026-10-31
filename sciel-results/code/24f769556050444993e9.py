import numpy as np
import time
import sys

# EVOLVE-BLOCK-START

# Global timing state
_start_time = None

# Weibull parameters for item size distribution
# Based on typical online bin packing tasks with capacity 100.
# Assuming items are not uniform but have a skew.
# Mean ~ 15-20.
# Weibull(k=2.0, lambda=20.0) -> Mean = 20 * Gamma(1.5) = 20 * 0.886 = 17.7
# Weibull(k=2.0, lambda=15.0) -> Mean = 15 * 0.886 = 13.3
# Let's use k=2.0, lambda=18.0 as a balanced guess.
_WEIBULL_K = 2.0
_WEIBULL_LAMBDA = 18.0
_CAPACITY = 100.0

def _weibull_survival(r):
    """
    Calculate the survival function S(r) = P(X > r) for Weibull(k, lambda).
    S(r) = exp(-(r / lambda)^k)
    """
    # Clip r to non-negative to avoid numerical issues, though r should be >= 0
    r_clipped = np.maximum(r, 0.0)
    # Optimization: if r is very large, S(r) is effectively 0
    # (r / lambda)^k
    z = (r_clipped / _WEIBULL_LAMBDA) ** _WEIBULL_K
    # exp(-z)
    return np.exp(-z)

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity for the arriving item.
    
    Strategy: Distribution-Aware Dead Space Minimization with Consolidation Bias.
    
    1. Calculate remaining capacity `r` for each bin if `item` is placed.
    2. Filter valid bins (r >= 0).
    3. For valid bins, compute a "Dead Space" penalty:
       Penalty = r * S(r)
       where S(r) is the probability that a future item is larger than r.
       This metric is low when r is very small (bin is full) or very large (bin is empty, 
       and S(r) is small). It is high for "middle" gaps that are likely to be wasted.
       We want to MINIMIZE this penalty.
    4. Add a Consolidation Bonus:
       Bonus = alpha * (current_fill / Capacity)
       We want to MAXIMIZE this to prefer bins that are already in use.
    
    Score = -Penalty + Bonus
    """
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
        
    # Deadline guard: 150s limit, keep 5s margin
    elapsed = time.time() - _start_time
    if elapsed > 145:
        # Fallback to simple Best Fit
        remaining = bins - item
        valid = remaining >= 0
        scores = np.full(len(bins), -np.inf)
        if np.any(valid):
            scores[valid] = -remaining[valid]
        return scores

    # Calculate remaining capacity if item is placed
    remaining = bins - item
    
    # Identify valid bins (enough capacity)
    valid = remaining >= 0
    
    # Initialize scores with -inf for invalid bins
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(valid):
        return scores
        
    valid_indices = np.where(valid)[0]
    new_rems = remaining[valid]
    current_fills = bins[valid]
    
    # Calculate Weibull Survival Function S(r)
    # S(r) = exp(-(r/lambda)^k)
    # We use this to estimate the "usefulness" of the remaining space.
    s_r = _weibull_survival(new_rems)
    
    # Dead Space Penalty: r * S(r)
    # This represents the expected "wasted" space if we assume the next item
    # is the only one that can fit in r. It's a proxy for fragmentation.
    dead_space = new_rems * s_r
    
    # Consolidation Bonus:
    # Prefer bins that are already more full.
    # Utilization = current_fill / Capacity
    # We want to maximize this.
    # Weight: How strong should this be?
    # If dead_space is ~ 10, and utilization is 0.5, bonus should be significant.
    # Let's try alpha = 10.0.
    alpha = 10.0
    consolidation_bonus = alpha * (current_fills / _CAPACITY)
    
    # Final Score
    # We want to minimize dead_space and maximize consolidation_bonus.
    scores[valid] = -dead_space + consolidation_bonus
    
    return scores

# EVOLVE-BLOCK-END
