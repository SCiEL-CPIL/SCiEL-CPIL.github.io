"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Probability of Improvement per unit cost.
    # This can be more robust than EI in high dimensions by encouraging exploration
    # of regions with high uncertainty, while still being cost-aware.
    z = (f_best - mu) / (sigma + 1e-9)
    pi = norm.cdf(z)
    return pi / cost
# EVOLVE-BLOCK-END
