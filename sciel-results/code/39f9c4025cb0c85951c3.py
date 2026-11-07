"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Expected Improvement with softer cost penalty.
    # Use 1/sqrt(cost) instead of 1/cost to allow exploration of expensive regions.
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    cost_safe = np.maximum(cost, eps)
    return ei / np.sqrt(cost_safe)
# EVOLVE-BLOCK-END
