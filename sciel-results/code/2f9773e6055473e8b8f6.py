"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Pure Lower Confidence Bound (LCB) with a fixed exploration constant.
    # We want to minimize, so lower predicted values are better.
    # LCB = mu - k * sigma
    # We return scores where higher means evaluate next, so we negate the LCB
    # or equivalently return -mu + k * sigma.
    # Using a high exploration constant to target high-uncertainty regions
    # where the global minimum might be hidden.
    
    # Clip sigma to avoid numerical issues, though it should be positive
    sigma = np.maximum(sigma, 1e-9)
    
    k = 3.0
    scores = -mu + k * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
