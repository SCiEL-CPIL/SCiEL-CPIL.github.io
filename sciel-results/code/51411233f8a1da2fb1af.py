"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected Improvement (EI) for minimization, with a small epsilon
    # to guard against zero/near-zero sigma values.
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), z = (f_best - mu) / sigma
    # Higher EI means the point is more likely to beat f_best.
    eps = 1e-9
    sigma_safe = np.where(sigma > eps, sigma, eps)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # Ensure all scores are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    return ei
# EVOLVE-BLOCK-END
