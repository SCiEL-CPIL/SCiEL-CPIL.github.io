"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is strictly positive for numerical stability
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add a small exploration term weighted by sigma
    # This encourages visiting regions with high uncertainty,
    # balancing the exploitation of EI with exploration of unknown areas.
    exploration_weight = 0.1
    exploration_term = exploration_weight * sigma
    
    # Cost-aware score
    # Divide by cost to penalize expensive evaluations
    score = (ei + exploration_term) / cost
    
    return score
# EVOLVE-BLOCK-END
