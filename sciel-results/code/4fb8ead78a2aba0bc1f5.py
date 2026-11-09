"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound for minimization
    # We want to find the minimum, so we want low mu.
    # We also want to explore where sigma is high.
    # Acquisition score: higher is better.
    # Score = -mu + beta * sigma
    
    beta = 1.0
    
    # Ensure sigma is non-negative and finite
    sigma_safe = np.where(np.isfinite(sigma) & (sigma >= 0), sigma, 0.0)
    
    # Calculate UCB score
    # Note: f_best is not used in standard UCB, but it's in the signature.
    # We ignore it for the score calculation.
    ucb_scores = -mu + beta * sigma_safe
    
    # Ensure finite scores
    ucb_scores = np.where(np.isfinite(ucb_scores), ucb_scores, -np.inf)
    
    return ucb_scores
# EVOLVE-BLOCK-END
