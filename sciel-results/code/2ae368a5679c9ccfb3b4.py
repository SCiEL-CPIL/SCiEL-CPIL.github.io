"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware EI + UCB exploration term
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost normalization: cost ranges from 1.0 to 2.0
    # We want to penalize high-cost points more heavily
    cost_penalty = 1.0 / cost
    
    # UCB exploration term: encourages sampling in high-uncertainty regions
    # Weight the exploration term to balance exploitation and exploration
    beta = 1.5
    ucb = beta * sigma_safe
    
    # Combine: (EI + UCB) / cost
    # The division by cost makes high-cost points less attractive
    score = (ei + ucb) * cost_penalty
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
