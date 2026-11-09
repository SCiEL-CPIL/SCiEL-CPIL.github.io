import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 300.0  # safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """Cost-aware acquisition function for Bayesian optimization of hartmann 3d."""
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5.0:
        # Return a simple fallback: negative mean divided by cost
        return -mu / np.maximum(cost, 1e-9)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-8)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    
    # Ensure cost is positive and finite
    cost = np.maximum(cost, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma
    # Clip z for numerical stability
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware: divide by cost
    scores = ei / cost
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
