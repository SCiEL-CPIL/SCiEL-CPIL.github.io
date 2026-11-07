"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected Improvement with optimism parameter xi
    # xi > 0 makes the acquisition function more exploratory
    xi = 0.1
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - xi - mu) / sigma_safe
    score = (f_best - xi - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return score
# EVOLVE-BLOCK-END
