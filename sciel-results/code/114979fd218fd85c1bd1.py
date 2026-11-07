import numpy as np
from scipy.stats import norm
import time

# Global timer for safety
_START_TIME = time.time()
_TIME_LIMIT = 300.0  # Safety margin below 330s

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using adaptive Cost-aware LCB.
    Returns scores of shape (2000,), where higher scores indicate better candidates.
    """
    # Check time budget
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        # Return a safe fallback if we're over time
        return np.zeros_like(mu)
    
    # Ensure safety in denominators
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Adaptive beta: scale by maximum sigma to balance exploration and exploitation
    max_sigma = np.max(sigma)
    beta = 2.0 * max_sigma
    
    # Lower Confidence Bound for minimization (lower is better)
    lcb = mu - beta * sigma_safe
    
    # Cost-aware LCB: penalize high-cost evaluations
    # We want to maximize the score, so we use -lcb / cost
    scores = -lcb / cost_safe
    
    # Ensure all outputs are finite
    scores = np.where(np.isfinite(scores), scores, -1e9)
    
    return scores
