"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Expected improvement per unit cost with clipping for stability.
    z = (f_best - mu) / (sigma + 1e-9)
    z = np.clip(z, -10, 10)
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    ei = (f_best - mu) * cdf + sigma * pdf
    # Avoid division by zero or negative cost
    cost = np.maximum(cost, 1e-6)
    return ei / cost
# EVOLVE-BLOCK-END
