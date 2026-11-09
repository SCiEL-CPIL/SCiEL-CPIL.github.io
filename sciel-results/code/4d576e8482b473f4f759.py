"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Use Lower Confidence Bound (UCB) for minimization
    # Minimize: mu - beta * sigma
    # Maximize (score): beta * sigma - mu
    
    # Beta controls exploration. 
    # A value around 1-3 is typical.
    beta = 2.0
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate score
    scores = beta * sigma - mu
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
