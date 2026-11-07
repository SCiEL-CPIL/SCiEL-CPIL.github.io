"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound for minimization.
    # Maximizes the score: -mu (exploitation) + beta * sigma (exploration).
    beta = 1.0
    return -mu + beta * sigma
# EVOLVE-BLOCK-END
