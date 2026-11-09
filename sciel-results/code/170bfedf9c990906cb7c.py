"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIMEOUT = 325.0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using UCB per unit cost.
    Score = (beta * sigma - mu) / cost.
    UCB provides optimistic exploration of high-uncertainty regions,
    structurally different from EI which only targets expected improvement.
    """
    if time.time() - _START_TIME > _TIMEOUT:
        return np.zeros(len(mu))

    n = len(mu)
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # UCB for minimization: maximize (-mu + beta * sigma)
    # beta controls exploration vs exploitation
    beta = 2.0
    ucb = beta * sigma_safe - mu
    
    # Cost-aware scoring: UCB / cost
    score = ucb / np.maximum(cost, 1e-9)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
