"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound for minimization.
    # We want to minimize f(x).
    # The optimistic lower bound is mu - beta * sigma.
    # We want to maximize the score, so we return the negative of the lower bound.
    # Score = -(mu - beta * sigma) = -mu + beta * sigma
    
    beta = 2.0
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
