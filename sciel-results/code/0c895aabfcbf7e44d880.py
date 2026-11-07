"""Rewritten acquisition for Bayesian optimization of sphere 1d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d.
    Uses Expected Improvement (EI) which is the standard and robust choice
    for minimizing a smooth function.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    fb = float(f_best)

    # Ensure sigma is strictly positive to avoid division by zero
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)

    # Standardized improvement
    z = (fb - mu) / sigma_safe

    # Expected Improvement for minimization
    # EI = (fb - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the CDF of the standard normal and phi is the PDF
    ei = (fb - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Ensure all scores are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)

    return ei
# EVOLVE-BLOCK-END
