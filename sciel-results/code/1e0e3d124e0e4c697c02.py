"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB)
    # We want to minimize f, so we maximize -mu + beta * sigma
    # beta controls exploration. 
    # A value around 1.0 to 2.0 is common.
    beta = 1.5
    return -mu + beta * sigma
# EVOLVE-BLOCK-END
