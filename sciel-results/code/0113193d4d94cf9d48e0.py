"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Pure GP-LCB acquisition: score = -mu - kappa * sigma
    # Higher score means more promising (lower expected objective with exploration).
    # Using kappa=1.5 for a balance between exploration and exploitation
    # suitable for the smooth 1D Styblinski-Tang function.
    kappa = 2.5
    return -mu - kappa * sigma
# EVOLVE-BLOCK-END
