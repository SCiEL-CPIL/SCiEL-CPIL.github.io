"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Upper Confidence Bound per unit cost.
    # For minimization, we want to maximize the potential for improvement:
    # f_best - (mu - kappa * sigma) = f_best - mu + kappa * sigma
    kappa = 1.0
    score = (f_best - mu + kappa * sigma) / cost
    return score
# EVOLVE-BLOCK-END
