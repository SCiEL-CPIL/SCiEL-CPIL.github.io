"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB) for minimization.
    # We want to minimize f, so we look for low mu and high sigma.
    # Score = beta * sigma - mu. Higher score is better.
    beta = 2.0
    return beta * sigma - mu
# EVOLVE-BLOCK-END
