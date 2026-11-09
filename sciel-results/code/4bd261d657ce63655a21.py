"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware expected improvement (EI per unit cost) with safeguards.
    # mu, sigma, f_best, cost are arrays of shape (N,)
    # Return N finite scores, higher = better.

    # Ensure finite inputs
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-12)

    # Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe

    # Use stable computation for large negative z
    # For z < -30, Phi(z) ~ 0 and phi(z) ~ 0, so EI ~ 0
    # For z > 30, Phi(z) ~ 1, phi(z) ~ 0, so EI ~ f_best - mu
    phi = norm.pdf(z)
    cdf = norm.cdf(z)

    ei = (f_best - mu) * cdf + sigma_safe * phi

    # Guard against non-finite values
    ei = np.where(np.isfinite(ei), ei, 0.0)

    # Cost-aware: divide by cost
    score = ei / cost_safe

    # Ensure all outputs are finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
