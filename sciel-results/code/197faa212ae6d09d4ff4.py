"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization
    # Score = -mu + beta * sigma
    # Higher score means more promising (lower expected value or higher uncertainty)
    beta = 1.5
    return -mu + beta * sigma
# EVOLVE-BLOCK-END
