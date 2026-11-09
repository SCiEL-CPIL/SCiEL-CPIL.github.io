"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Ensure sigma is strictly positive to avoid division by zero or numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Adaptive UCB strategy
    # We maximize: -mu + beta * sigma
    # Beta is adapted based on f_best.
    # High f_best (early stage) -> High beta (more exploration)
    # Low f_best (late stage) -> Low beta (more exploitation)
    
    # Map f_best [0, 25] to beta [0.1, 2.0]
    # Linear scaling: beta = 0.1 + (2.0 - 0.1) * (f_best / 25.0)
    # If f_best is 25, beta = 2.1 -> clip to 2.0
    # If f_best is 0, beta = 0.1
    
    beta = 0.1 + 1.9 * (f_best / 25.0)
    beta = np.clip(beta, 0.1, 2.0)
    
    # Score for minimization: we want low mu and high sigma
    # So we maximize -mu + beta * sigma
    score = -mu + beta * sigma
    
    return score
# EVOLVE-BLOCK-END
