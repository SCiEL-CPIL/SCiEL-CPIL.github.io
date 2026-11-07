"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected improvement with a slightly reduced exploration term.
    # This tunes the balance between exploration and exploitation within the EI family.
    z = (f_best - mu) / (sigma + 1e-9)
    lambda_exploration = 0.9
    return (f_best - mu) * norm.cdf(z) + lambda_exploration * sigma * norm.pdf(z)
# EVOLVE-BLOCK-END
