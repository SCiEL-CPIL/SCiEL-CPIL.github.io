"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization.
    # Score = -mu + beta * sigma
    # Higher score indicates a more promising candidate (lower mean or higher uncertainty).
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-10)
    
    # Exploration parameter
    beta = 2.0
    
    # UCB score for minimization
    # We want to minimize f, so we prefer low mu.
    # We add beta * sigma to encourage exploration of uncertain regions.
    # The score to maximize is: -mu + beta * sigma
    ucb_score = -mu + beta * sigma
    
    # Ensure output is finite
    ucb_score = np.nan_to_num(ucb_score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return ucb_score
# EVOLVE-BLOCK-END
