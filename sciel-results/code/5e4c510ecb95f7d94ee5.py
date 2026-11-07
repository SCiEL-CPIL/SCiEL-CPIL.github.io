"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware LCB: score = (beta * sigma - mu) / cost
    # LCB balances exploitation (low mu) and exploration (high sigma)
    # while penalizing expensive evaluations via division by cost
    
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Dynamic beta: start aggressive for exploration, decay over time
    # Use a fixed beta tuned for Rastrigin 2D with Matern-5/2 GP
    beta = 2.5
    
    # LCB score: higher is better (we want low mu, high sigma, low cost)
    score = (beta * sigma_safe - mu) / cost_safe
    
    # Ensure finite outputs
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
