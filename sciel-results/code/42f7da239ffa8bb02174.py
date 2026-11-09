"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected Improvement with slight exploration boost
    # Using a modified EI that adds a small exploration term
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Standard EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a small exploration bonus based on uncertainty
    # This helps escape local optima in the early stages
    exploration_weight = 0.1
    exploration_bonus = exploration_weight * sigma_safe
    
    return ei + exploration_bonus
# EVOLVE-BLOCK-END
