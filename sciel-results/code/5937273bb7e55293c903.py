"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization
    # We maximize the score to select the next candidate.
    # For minimization, we prefer low mu (exploitation) and high sigma (exploration).
    # Score = -mu + beta * sigma
    
    # Choose beta to balance exploration and exploitation.
    # A value around 1.0 to 2.0 is typical.
    # Since Ackley is multimodal and we have a small budget, we favor exploration slightly.
    beta = 1.5
    
    # Ensure sigma is non-negative, though it should be by definition
    sigma = np.maximum(sigma, 0.0)
    
    scores = -mu + beta * sigma
    
    # Return finite scores
    return scores
# EVOLVE-BLOCK-END
