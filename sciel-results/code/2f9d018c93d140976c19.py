"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 300.0
_MARGIN = 10.0

def _remaining_time():
    return _TIME_LIMIT - (time.time() - _start_time) - _MARGIN

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using multiplicative Probability of Improvement
    with sigma boost, divided by cost.
    
    PI(x) = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    The multiplicative form PI * (1 + beta * sigma) couples exploitation and exploration.
    Dividing by cost makes it cost-aware.
    """
    # Deadline guard
    if _remaining_time() < 1.0:
        # Fast fallback
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        pi = norm.cdf(z)
        beta = 0.5
        scores = pi * (1.0 + beta * sigma_safe)
        cost_safe = np.maximum(cost, 1e-8)
        scores = scores / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)

    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute Probability of Improvement (PI)
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    
    # Multiplicative exploration boost
    # beta controls how much we boost uncertain regions
    beta = 0.3
    scores = pi * (1.0 + beta * sigma_safe)
    
    # Cost safety
    cost_safe = np.maximum(cost, 1e-8)
    
    # Cost-aware: divide by cost
    scores = scores / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
