"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Exploitation-focused acquisition: Probability of Improvement weighted by
    # the magnitude of the improvement and inverse cost.
    # This prioritizes regions with both high confidence and high potential gain,
    # avoiding the exploration bonus of standard EI.
    
    # Ensure sigma is not zero to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-scores
    z = (f_best - mu) / sigma_safe
    
    # Calculate Probability of Improvement
    pi = norm.cdf(z)
    
    # Calculate the expected improvement magnitude (clipped at 0)
    # This term ensures we favor points with significantly lower mean predictions
    improvement = np.maximum(f_best - mu, 0.0)
    
    # Weight by inverse cost and improvement magnitude
    scores = improvement * pi / cost
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
