"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIMEOUT = 325.0  # Safety margin before 330s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement with
    adaptive exploration based on uncertainty and cost efficiency.
    """
    # Deadline guard: if we're too close to timeout, return simple EI/cost
    if time.time() - _START_TIME > _TIMEOUT:
        sigma = np.clip(sigma, 1e-8, None)
        z = (f_best - mu) / sigma
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        score = ei / cost
        return np.where(np.isfinite(score), score, 0.0)
    
    # Clip sigma to avoid division by zero
    sigma = np.clip(sigma, 1e-8, None)
    
    # Compute Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add exploration bonus proportional to uncertainty
    # Weight it to balance exploitation and exploration
    exploration_weight = 0.05
    exploration = exploration_weight * sigma
    
    # Cost-aware scoring: divide by cost to prefer cheaper points
    # Add a small regularization to avoid division by very small costs
    score = (ei + exploration) / (cost + 1e-6)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
