"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization.
    # We maximize the score, which corresponds to the most promising point.
    # For minimization, promising means low mean (mu) and high uncertainty (sigma).
    # Score = kappa * sigma - mu
    # A kappa of 2.0 is a standard choice for balancing exploration and exploitation.
    kappa = 2.0
    scores = kappa * sigma - mu
    return scores
# EVOLVE-BLOCK-END
