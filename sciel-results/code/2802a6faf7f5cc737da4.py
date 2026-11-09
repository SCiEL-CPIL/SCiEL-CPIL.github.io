"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected improvement for minimization.
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues with extreme values
    z = np.clip(z, -50.0, 50.0)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # Ensure finite scores
    ei = np.where(np.isfinite(ei), ei, 0.0)
    return ei
# EVOLVE-BLOCK-END
