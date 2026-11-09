"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Standard Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Exploration bonus: reward high uncertainty
    # This creates a hybrid between EI (exploitation) and a pure exploration term
    exploration_weight = 0.1
    score = ei + exploration_weight * sigma
    
    return score
# EVOLVE-BLOCK-END
