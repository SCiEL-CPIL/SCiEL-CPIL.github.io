"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Expected improvement per unit cost with a small noise parameter
    # to balance exploration and exploitation
    xi = 0.05 * max(abs(f_best), 1.0)
    z = (f_best - xi - mu) / (sigma + 1e-9)
    ei = (f_best - xi - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    return ei / cost
# EVOLVE-BLOCK-END
