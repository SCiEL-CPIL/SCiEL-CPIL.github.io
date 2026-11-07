"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # UCB for minimization: maximize -mu + beta * sigma
    # Higher beta encourages more exploration
    beta = 1.5
    return -mu + beta * sigma
# EVOLVE-BLOCK-END
