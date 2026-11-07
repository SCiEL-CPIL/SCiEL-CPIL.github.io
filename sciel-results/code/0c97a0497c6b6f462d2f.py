"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import erf


def acquisition(mu, sigma, f_best, cost):
    # Probability of Improvement (PI) weighted by inverse cost
    # Using erf for efficiency and stability
    # PI(x) = 0.5 * (1 + erf((f_best - mu) / (sigma * sqrt(2))))
    
    # Ensure sigma is not zero to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-scores for erf
    # z_erf = (f_best - mu) / (sigma * sqrt(2))
    sqrt2 = np.sqrt(2.0)
    z_erf = (f_best - mu) / (sigma_safe * sqrt2)
    
    # Calculate Probability of Improvement
    pi = 0.5 * (1.0 + erf(z_erf))
    
    # Weight by inverse cost
    # Using a softer penalty: 1 / (1 + cost) to reduce bias against high-cost regions
    # while still respecting the cost-awareness requirement
    scores = pi / (1.0 + cost)
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
