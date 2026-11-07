import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 325.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Optimistic LCB per cost.
    Returns scores of shape (2000,), where higher scores indicate better candidates.
    """
    # Explicit deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback: Simple Cost-Aware LCB
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        # Optimistic score: f_best - mu + beta * sigma, divided by cost
        beta = 5.0
        scores = (f_best - mu + beta * sigma_safe) / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)

    # Ensure safety in denominators
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Cost-aware Optimistic LCB
    # We want to maximize: (Potential Improvement + Exploration Bonus) / Cost
    # Potential Improvement relative to f_best: f_best - mu
    # Exploration Bonus: beta * sigma
    beta = 5.0
    scores = (f_best - mu + beta * sigma_safe) / cost_safe
    
    # Ensure all outputs are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
