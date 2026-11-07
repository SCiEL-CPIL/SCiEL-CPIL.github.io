"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIMEOUT = 320.0  # Safety margin before 330s kill


def acquisition(mu, sigma, f_best):
    # Time guard: if we're close to timeout, return a safe default
    elapsed = time.time() - _START_TIME
    if elapsed > _TIMEOUT:
        return np.ones_like(mu) * 0.0

    # Numerically stable Expected Improvement
    # z = (f_best - mu) / sigma, where sigma > 0
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe

    # For large positive z, norm.cdf(z) ≈ 1 and norm.pdf(z) ≈ 0
    # For large negative z, norm.cdf(z) ≈ 0 and norm.pdf(z) ≈ 0
    # Use stable computation:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is CDF and phi is PDF of standard normal

    # Clip z to avoid overflow in exp for very large |z|
    z_clipped = np.clip(z, -50, 50)

    # Compute CDF and PDF
    cdf_vals = norm.cdf(z_clipped)
    pdf_vals = norm.pdf(z_clipped)

    # Expected Improvement (for minimization, we want to improve below f_best)
    # The improvement is max(0, f_best - f(x)), so EI = E[max(0, f_best - f(x))]
    # = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * cdf_vals + sigma_safe * pdf_vals

    # Ensure finite values
    ei = np.where(np.isfinite(ei), ei, 0.0)

    return ei
# EVOLVE-BLOCK-END
