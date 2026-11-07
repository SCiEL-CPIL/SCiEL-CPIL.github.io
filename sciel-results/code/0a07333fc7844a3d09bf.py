"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB) for minimization.
    # We want to minimize mu(x) - beta * sigma(x).
    # The acquisition function returns scores where higher is better.
    # So we return beta * sigma(x) - mu(x).
    
    # Choose a beta that balances exploration and exploitation.
    # A value of 1.5 is a common choice for moderate exploration.
    beta = 1.5
    
    # Calculate LCB scores
    scores = beta * sigma - mu
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
