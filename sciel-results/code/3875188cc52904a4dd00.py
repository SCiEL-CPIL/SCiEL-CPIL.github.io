"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track start time for deadline guard
_START_TIME = None
_DEADLINE_MARGIN = 5.0  # seconds before the 150s limit
_MAX_EVAL_TIME = 150.0 - _DEADLINE_MARGIN

# Weibull parameters
# We adapt the scale parameter lambda based on the observed mean item size.
# We assume a shape parameter k. Let's use k=2.0.
# Mean = lambda * Gamma(1 + 1/k)
# For k=2, Gamma(1.5) = sqrt(pi)/2
_WEIBULL_K = 2.0
_GAMMA_TERM = np.sqrt(np.pi) / 2.0  # Gamma(1.5)

# State for online estimation
_ITEM_SUM = 0.0
_ITEM_COUNT = 0
_ESTIMATED_LAMBDA = 15.0  # Initial guess

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Adaptive Distribution-aware "Expected Dead Space" minimization.
    
    We adapt the Weibull scale parameter (lambda) to the observed mean item size.
    We use a score that penalizes remainders that are "medium" sized (most likely to
    become dead space) and rewards remainders that are very small (tight fit) or
    very large (safe gap for future items).
    
    Score = - (r * exp(-(r/lambda_est)^k))
    """
    global _START_TIME
    global _ITEM_SUM
    global _ITEM_COUNT
    global _ESTIMATED_LAMBDA
    
    # Initialize timing if not started
    if _START_TIME is None:
        _START_TIME = time.time()
        
    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _MAX_EVAL_TIME:
        # Fallback to simple Best Fit to ensure completion
        remaining = bins - item
        valid = remaining >= 0
        scores = np.full(len(bins), -np.inf)
        if np.any(valid):
            scores[valid] = -remaining[valid]
        return scores

    # Update online statistics
    _ITEM_SUM += item
    _ITEM_COUNT += 1
    
    # Calculate current mean
    current_mean = _ITEM_SUM / _ITEM_COUNT
    
    # Estimate lambda
    new_lambda = current_mean / _GAMMA_TERM
    _ESTIMATED_LAMBDA = new_lambda
    
    # Calculate remaining capacity
    remaining = bins - item
    
    # Identify valid bins (enough capacity)
    valid = remaining >= 0
    
    # Initialize scores with -inf for invalid bins
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(valid):
        return scores
        
    valid_rems = remaining[valid]
    
    # Compute the "Dead Space" metric
    r = np.maximum(valid_rems, 0.0)
    
    # Use the estimated lambda
    lambda_est = _ESTIMATED_LAMBDA
    if lambda_est < 1e-6:
        lambda_est = 1e-6

    ratio = r / lambda_est
    power_term = np.power(ratio, _WEIBULL_K)
    survival = np.exp(-power_term)
    
    dead_space = r * survival
    
    # Score: minimize dead space
    # We do NOT add a linear penalty beta*r.
    # This creates a preference for r=0 (tight) and r->infinity (open).
    # This avoids "half-full" bins which are prone to fragmentation.
    scores[valid] = -dead_space
    
    return scores
# EVOLVE-BLOCK-END
