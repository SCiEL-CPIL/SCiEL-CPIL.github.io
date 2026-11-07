import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 320.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Weighted UCB.
    Higher score = more likely to be evaluated.
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Fallback: exploit best mean with cost penalty
        scores = -mu / cost
        if not np.all(np.isfinite(scores)):
            scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    # Ensure sigma is non-negative
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    
    # Adaptive Beta:
    # Start with high exploration (beta ~ 2.0) when f_best is large (poor).
    # Reduce exploration (beta ~ 1.0) when f_best is small (good).
    # Using a soft transition based on f_best.
    # Assuming f_best > 0.
    # If f_best is 0, we are done, but we still need to return scores.
    # Scale: f_best / (f_best + 1) goes 0->1 as f_best 0->inf.
    # We want beta high when f_best is large.
    # So beta = 1.0 + 1.0 * (f_best / (f_best + 1.0))
    # If f_best = 0, beta = 1.0.
    # If f_best = 100, beta ~ 2.0.
    
    # Handle case where f_best might be negative? Rosenbrock min is 0.
    # So f_best >= 0.
    
    beta = 1.0 + 1.0 * (f_best / (f_best + 1.0))
    
    # UCB for minimization: -mu + beta * sigma
    # We maximize this value.
    ucb = -mu + beta * sigma_safe
    
    # Cost-weighted
    # cost is in [1, 2]
    scores = ucb / cost
    
    # Safety check for finite values
    if not np.all(np.isfinite(scores)):
        scores = np.where(np.isfinite(scores), scores, 0.0)
        
    return scores
