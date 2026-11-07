"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 310.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement with
    a cost-adjusted exploration term.
    
    mu, sigma, f_best, cost: arrays of shape (2000,)
    Returns: array of 2000 finite scores (higher = better candidate to evaluate)
    """
    # Deadline guard: if we've spent too much time, return a simple fallback
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        return np.zeros(len(mu), dtype=np.float64)

    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip EI to be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-awareness:
    # Instead of a hard division by cost (range 1-2), we use a softer penalty.
    # cost is in [1, 2].
    # cost_factor = 1 + 0.5 * (cost - 1) -> range [1, 1.5]
    # This reduces the bias against the center (cost 1.5) compared to dividing by 1.5 vs 1.0.
    cost_factor = 1.0 + 0.5 * (cost - 1.0)
    
    # Exploration term coupled with cost penalty.
    # High uncertainty (sigma) encourages evaluation, but is discounted by cost.
    # beta controls the weight of uncertainty relative to expected improvement.
    beta = 0.1
    
    # Combined score
    scores = (ei + beta * sigma_safe) / cost_factor
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
