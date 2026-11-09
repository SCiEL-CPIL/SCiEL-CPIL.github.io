"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB) for minimization.
    # We want to minimize f, so we maximize -mu + kappa * sigma.
    # kappa=1.0 balances exploration and exploitation.
    kappa = 1.0
    scores = -mu + kappa * sigma
    return scores
# EVOLVE-BLOCK-END
