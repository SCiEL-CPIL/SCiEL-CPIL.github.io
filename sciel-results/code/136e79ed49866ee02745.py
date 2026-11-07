# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_T0 = time.time()
_DEADLINE = _T0 + 315.0  # safety margin vs 330s kill


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of 1D sphere function.
    Uses Expected Improvement (EI) which is well-suited for 1D problems
    where the GP posterior is informative. EI balances exploration and
    exploitation optimally for this setting.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    fb = float(f_best)

    # Ensure sigma is positive for stable computation
    sigma_safe = np.maximum(sigma, 1e-8)

    # Compute Expected Improvement (EI)
    # EI(x) = (fb - mu) * Phi(z) + sigma * phi(z)
    # where z = (fb - mu) / sigma, Phi is standard normal CDF, phi is PDF
    z = (fb - mu) / sigma_safe
    ei = (fb - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Clip EI to be non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)

    # Ensure finite values
    ei = np.where(np.isfinite(ei), ei, 0.0)

    # If time is running out, return immediately
    if time.time() > _DEADLINE:
        return ei

    return ei
# EVOLVE-BLOCK-END
