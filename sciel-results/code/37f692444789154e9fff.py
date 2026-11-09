"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: exponential penalty to favor cheaper points
    # cost is in [1, 2], so lambda controls how aggressively we penalize high cost
    # lambda=1 gives strong preference for lower-cost regions
    lambda_cost = 1.0
    cost_penalty = np.exp(-lambda_cost * (cost - 1.0))
    
    # Combine exploitation (EI with cost penalty) and exploration
    # Exploration term uses UCB-like approach with uncertainty
    kappa = 0.1  # exploration parameter
    explore_term = kappa * sigma_safe
    
    # Final score: cost-weighted EI + exploration
    score = ei * cost_penalty + explore_term * cost_penalty
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
