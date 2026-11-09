"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate the standardized improvement z-score
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Add an exploration bonus (lambda) to encourage exploration of high uncertainty regions
    # This shifts the threshold, effectively computing P(f(x) < f_best + lambda * sigma)
    lambda_explore = 1.0
    z_adjusted = z + lambda_explore
    
    # Probability of Improvement with exploration bonus
    scores = norm.cdf(z_adjusted)
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
