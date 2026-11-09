"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # UCB (Upper Confidence Bound) with cost awareness.
    # UCB encourages exploration (high sigma) and exploitation (low mu).
    # Dividing by cost penalizes expensive evaluations.
    # Beta = 2.0 balances exploration and exploitation for the Levy function.
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate UCB
    beta = 2.0
    ucb = mu + beta * sigma
    
    # Cost-aware score
    # Cost is in [1, 2]. 
    # Using division is standard.
    score = ucb / cost
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
