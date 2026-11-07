"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    # UCB (Upper Confidence Bound) acquisition function for minimization.
    # We want to maximize the score, so we select points with low mean (mu) 
    # and high uncertainty (sigma).
    # score = -mu + beta * sigma
    
    # Beta controls the exploration/exploitation trade-off.
    # Based on inspirations, a high beta (e.g., 4.0) is effective for 
    # multimodal functions with limited evaluations.
    beta = 4.0
    
    # Ensure sigma is positive to avoid issues, though GP sigma should be > 0
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute UCB scores
    scores = -mu + beta * sigma
    
    return scores
# EVOLVE-BLOCK-END
