# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a linear regret lower bound.
    
    Formula: (f_best - mu + 0.5 * sigma) / cost
    
    This directly optimizes the expected reduction in simple regret per unit cost.
    It's a linear approximation that balances exploration (via sigma) and exploitation
    (via f_best - mu) while being cost-aware.
    """
    # Linear regret lower bound: expected improvement approximation
    # f_best - mu is the exploitation term (how much better than current best)
    # 0.5 * sigma is the exploration term (uncertainty)
    # Dividing by cost gives us regret reduction per unit cost
    
    # Ensure sigma is positive and handle edge cases
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute the acquisition score
    score = (f_best - mu + 0.5 * sigma) / cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
