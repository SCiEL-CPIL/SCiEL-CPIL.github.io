"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Pure GP-LCB acquisition: score = -mu - kappa * sigma
    # Higher score means more promising (lower expected objective with exploration).
    kappa = 2.0
    return -mu - kappa * sigma
# EVOLVE-BLOCK-END
