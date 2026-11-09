"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Expected Improvement with exploration bonus.
    # Use a small epsilon to avoid division by zero.
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware: divide by cost to prefer cheaper points
    # Add a small exploration bonus for high uncertainty
    # The cost is in [1, 2], so 1/cost is in [0.5, 1]
    # We want to balance exploitation (EI) and exploration (sigma)
    
    # Use a weighted combination: EI/cost + small exploration term
    # The exploration term helps when EI is similar but uncertainty is high
    exploration_weight = 0.1
    score = ei / cost + exploration_weight * sigma / cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
