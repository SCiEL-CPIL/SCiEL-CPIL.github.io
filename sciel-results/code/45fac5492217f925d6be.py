"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0  # safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() - _start_time > _DEADLINE:
        # Return a deterministic fallback if we're over deadline
        # Use cost-aware EI as fallback
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        score = ei / cost_safe
        return np.where(np.isfinite(score), score, 0.0)
    
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # UCB component for exploration
    beta = 1.0
    ucb = -mu + beta * sigma
    
    # Blend: weight EI and UCB, both cost-aware
    alpha = 0.5
    score = (alpha * ei + (1 - alpha) * ucb) / cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
