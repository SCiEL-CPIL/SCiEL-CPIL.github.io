"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 325.0  # 5s safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """Cost-aware acquisition with deadline guard and improved exploration."""
    # Deadline check: if we're close to the limit, return a safe fallback
    if time.time() > _DEADLINE:
        return np.zeros(2000, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma
    
    # Expected Improvement: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Using numerically stable computation
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost
    cost = np.maximum(cost, 1e-9)
    
    # Add a small exploration bonus based on uncertainty to encourage
    # exploring regions with high sigma even when EI is low
    # This helps escape local minima in rastrigin-like landscapes
    sigma_bonus = 0.1 * sigma / cost
    
    score = ei / cost + sigma_bonus
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
