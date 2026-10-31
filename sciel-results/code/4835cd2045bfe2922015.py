"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for deadline guard
_start_time = None

# Weibull distribution parameters for item sizes
# These are heuristic estimates for typical online bin packing with capacity 100.
# k=2.0 (shape), lambda=12.0 (scale)
# This gives mean ~10.6, which is reasonable for small-to-medium items.
_WEIBULL_K = 2.0
_WEIBULL_LAMBDA = 12.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin. The item goes to the highest-scoring bin.
    
    Strategy: Combine the 'Dead Space' insight (avoiding medium-sized remainders 
    that are too small for typical future items but too large to be negligible) 
    with a consolidation bias.
    
    The core idea is that a remainder `r` has an expected "waste" or "dead space" 
    that is non-monotonic. Very small `r` is good (tight fit). Very large `r` is 
    bad (fragmentation), but medium `r` can be particularly bad if it falls in a 
    'gap' where most future items are larger than `r`.
    
    We model the penalty for a remainder `r` as:
    1. Dead Space: r * S(r), where S(r) is the survival function P(X > r).
       This term is high when `r` is medium-sized and many future items are larger.
       It is low when `r` is very small (tight fit) or very large (S(r) ~ 0).
    2. Consolidation Bias: A small linear penalty on `r` to encourage using 
       smaller bins when dead space is comparable, reducing the number of open bins.
    
    We subtract this penalty from a base score to get the final score (higher is better).
    """
    global _start_time
    
    # Initialize timing if not started
    if _start_time is None:
        _start_time = time.time()
        
    # Deadline guard: keep 5s margin before 150s limit
    elapsed = time.time() - _start_time
    if elapsed > 145:
        # Fallback to simple Best Fit
        remaining = bins - item
        valid = remaining >= 0
        scores = np.full(len(bins), -np.inf)
        if np.any(valid):
            scores[valid] = -remaining[valid]
        return scores

    # Calculate remaining capacity
    remaining = bins - item
    
    # Identify valid bins (enough capacity)
    valid = remaining >= 0
    
    # Initialize scores with -inf for invalid bins
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(valid):
        return scores
        
    valid_indices = np.where(valid)[0]
    new_rems = remaining[valid]
    
    # Compute survival function: S(r) = exp(-(r/lambda)^k)
    # We use the estimated Weibull parameters.
    scaled = new_rems / _WEIBULL_LAMBDA
    survival = np.exp(-np.power(scaled, _WEIBULL_K))
    
    # Dead Space Penalty:
    # This term is high when `r` is in the "danger zone" where it's too small for 
    # most future items (high S(r)) but not small enough to be negligible (r > 0).
    dead_space = new_rems * survival
    
    # Consolidation Bias:
    # A small linear term to prefer smaller remainders, breaking ties in dead space
    # and generally encouraging bin closure.
    consolidation = 0.05 * new_rems
    
    # Total penalty
    penalty = dead_space + consolidation
    
    # Score is negative penalty (higher is better)
    scores_valid = -penalty
    
    # Small tie-breaker: prefer lower index for stability
    scores_valid -= np.arange(len(valid_indices)) * 1e-10
    
    scores[valid] = scores_valid
    
    return scores

# EVOLVE-BLOCK-END
