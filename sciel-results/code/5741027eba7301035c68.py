"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # UCB acquisition: mu - beta * sigma (for minimization)
    # Higher score means more likely to be evaluated next
    # For minimization, we want low mu and high sigma
    # So score = -mu + beta * sigma
    beta = 1.0
    return -mu + beta * sigma
# EVOLVE-BLOCK-END
