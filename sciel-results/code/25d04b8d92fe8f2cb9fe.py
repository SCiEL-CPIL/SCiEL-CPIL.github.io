"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Lower Confidence Bound (LCB) for minimization.
    # We want to minimize f(x) and cost(x).
    # Objective: Minimize LCB(x) + gamma * cost(x)
    # Equivalent to Maximize: -mu(x) + beta * sigma(x) - gamma * cost(x)
    
    # Adaptive exploration parameter beta
    # Scale with the maximum uncertainty in the current candidate set
    max_sigma = np.max(sigma)
    # Use a factor of 2.0 for ~95% confidence interval width scaling
    beta = 2.0 * max_sigma
    
    # Cost penalty gamma
    # Cost ranges from 1.0 to 2.0. Function values range roughly -0.8 to 1.0.
    # A penalty of 1.0 per unit cost balances exploration/exploitation with cost.
    gamma = 1.0
    
    # Calculate score
    score = -mu + beta * sigma - gamma * cost
    
    # Ensure finite scores, replace non-finite with very negative values
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END
