"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Expected improvement per unit cost.
    # Add a small floor to sigma to encourage exploration in regions
    # where the GP might be overconfident (low variance).
    sigma = np.maximum(sigma, 0.05)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    return ei / cost
# EVOLVE-BLOCK-END
