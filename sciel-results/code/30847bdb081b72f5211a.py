"""Combined program: Distribution-aware Best Fit with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for deadline guard
_start_time = None

# Weibull parameters for the item distribution
# Based on typical online bin packing with capacity 100
# k=2.0 (shape), lambda=15.0 (scale)
# This gives mean ~8.8, which is reasonable for small-to-medium items
_WEIBULL_K = 2.0
_WEIBULL_LAMBDA = 15.0


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    global _start_time
    
    # Initialize timing if not started
    if _start_time is None:
        _start_time = time.time()
        
    # Deadline guard: 150s limit, keep 5s margin
    elapsed = time.time() - _start_time
    if elapsed > 145:
        # Fallback to simple Best Fit to ensure completion
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
    
    # Distribution-aware scoring:
    # We use the "Dead Space" metric: r * exp(-(r/lambda)^k)
    # This creates a valley (high score) at r=0 and r->infinity,
    # with a peak (low score) at some intermediate r.
    # This encourages either perfect fit or very large remainders,
    # penalizing the "dead zone" where space is wasted.
    
    # For Weibull(k, lambda):
    # S(r) = exp(-(r/lambda)^k)
    # Dead space metric = r * S(r) = r * exp(-(r/lambda)^k)
    
    # We want to MINIMIZE dead space, so we negate it for the score
    # (higher score = better bin)
    
    # Compute the dead space for each valid bin
    r = new_rems
    # Avoid numerical issues: if r is very large, exp(-(r/lambda)^k) -> 0
    # If r is 0, the product is 0
    dead_space = r * np.exp(-(r / _WEIBULL_LAMBDA) ** _WEIBULL_K)
    
    # Score = -dead_space (we want to minimize dead space, so higher score is better)
    scores[valid] = -dead_space
    
    return scores
# EVOLVE-BLOCK-END
