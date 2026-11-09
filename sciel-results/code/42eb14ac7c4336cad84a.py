"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization.
    # We maximize the score: -mu + beta * sigma.
    # This balances exploitation (low mu) and exploration (high sigma).
    beta = 1.5
    return -mu + beta * sigma
# EVOLVE-BLOCK-END
