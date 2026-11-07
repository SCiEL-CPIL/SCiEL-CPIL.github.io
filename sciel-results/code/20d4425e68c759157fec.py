"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB)
    # We want to minimize f(x), so we look for low mu - beta * sigma.
    # The score returned should be higher for better candidates.
    # So Score = - (mu - beta * sigma) = -mu + beta * sigma.
    beta = 2.5
    scores = -mu + beta * sigma
    return scores
# EVOLVE-BLOCK-END
