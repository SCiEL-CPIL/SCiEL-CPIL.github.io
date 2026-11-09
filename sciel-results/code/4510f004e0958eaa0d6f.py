"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization.
    # We want to minimize mu(x) - beta * sigma(x).
    # Therefore, we maximize the score: beta * sigma(x) - mu(x).
    # This approach balances exploitation (low mu) and exploration (high sigma).
    
    beta = 2.0
    scores = beta * sigma - mu
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
