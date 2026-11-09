"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # UCB with beta = 0.5
    # Higher score means more likely to be evaluated
    # Minimization: we want low mu, high sigma (exploration)
    # Score = -mu + beta * sigma
    beta = 0.5
    return -mu + beta * sigma
# EVOLVE-BLOCK-END
