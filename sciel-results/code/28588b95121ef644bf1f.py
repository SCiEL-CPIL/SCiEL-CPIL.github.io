"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Pure GP-LCB: score = -mu - kappa * sigma
    # For minimization, we want to find the point with the lowest possible value.
    # The lower confidence bound is mu - kappa * sigma.
    # We return the negative of the LCB so that higher scores correspond to lower objective values.
    # score = -(mu - kappa * sigma) = -mu + kappa * sigma
    
    kappa = 2.0
    score = -mu + kappa * sigma
    
    return score
# EVOLVE-BLOCK-END
