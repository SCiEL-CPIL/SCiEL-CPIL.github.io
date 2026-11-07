"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected Improvement (EI) with an added exploration term
    # Based on findings that EI + 0.1*sigma performs well (score ~1.28)
    
    # Safe sigma for division
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard EI calculation
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add exploration component to balance exploitation
    exploration_weight = 0.1
    scores = ei + exploration_weight * sigma
    
    # Ensure outputs are finite
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    
    return scores
# EVOLVE-BLOCK-END
