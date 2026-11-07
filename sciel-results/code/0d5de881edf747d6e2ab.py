"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Use Upper Confidence Bound (UCB) for minimization.
    # Score = -mu + beta * sigma
    # Higher score indicates a more promising candidate.
    beta = 1.5
    score = -mu + beta * sigma
    return score
# EVOLVE-BLOCK-END
