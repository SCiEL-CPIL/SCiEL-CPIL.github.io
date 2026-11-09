"""Acquisition function for Bayesian optimization of branin 2d.
Uses Expected Improvement (EI) for a principled balance of exploration
and exploitation."""
import time
import numpy as np
from scipy.stats import norm

# Global deadline guard (330 s wall-clock limit; keep a safety margin).
_START = time.time()
_DEADLINE = _START + 325.0  # 5 s safety margin


def _elapsed():
    return time.time() - _START


def _is_dead():
    return _elapsed() > _DEADLINE


def acquisition(mu, sigma, f_best):
    """Return 2000 finite scores (higher = evaluate next).

    mu, sigma: shape (2000,), sigma is std-dev in objective units.
    f_best: lowest observed objective value so far.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = mu.shape[0]

    # Guard against non-finite inputs
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)

    # Deadline guard: if we're over time, just return a deterministic
    # exploitation score (negative mu) to keep the loop moving.
    if _is_dead():
        return -mu.copy()

    # Compute Expected Improvement (EI).
    # EI is the expected reduction in the objective value.
    # Higher EI means more promising.
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Ensure finite scores
    ei = np.where(np.isfinite(ei), ei, 0.0)

    return ei
