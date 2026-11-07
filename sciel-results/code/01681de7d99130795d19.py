"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization.
    # We maximize the score, which corresponds to the optimistic lower bound.
    # Lower bound = mu - beta * sigma
    # Score = -Lower Bound = beta * sigma - mu
    
    # A beta of 1.5 balances exploration and exploitation.
    # This is a structural change from the non-linear EI to a linear UCB.
    beta = 1.5
    return beta * sigma - mu
# EVOLVE-BLOCK-END
