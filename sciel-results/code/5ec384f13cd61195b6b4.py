"""Cost-aware acquisition for Griewank-2D BO with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Wall-clock deadline guard (330 s kill, use 325 s safety margin)
_T0 = time.time()
_DEADLINE = 325.0


def _remaining():
    """Return seconds remaining before deadline; 0 if exceeded."""
    return max(0.0, _DEADLINE - (time.time() - _T0))


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition score for minimizing Griewank-2D.

    Combines expected improvement (EI) with a local uncertainty bonus
    (exploration), all normalized by cost.

    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP posterior mean in objective units.
    sigma : ndarray, shape (2000,)
        GP posterior std in objective units.
    f_best : float
        Best (lowest) observed objective value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate.

    Returns
    -------
    ndarray, shape (2000,)
        Finite scores; higher = more worth evaluating next.
    """
    # Guard: if we're near deadline, return a safe finite array immediately
    if _remaining() < 0.5:
        return np.zeros_like(mu, dtype=np.float64)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Clamp sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)

    # Expected Improvement (minimization)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Uncertainty bonus: encourage exploring high-uncertainty regions
    # Scale by a small factor relative to the typical EI scale
    # Use a normalized uncertainty term: sigma / (f_best + 1e-9) to be scale-aware
    # But keep it modest so EI dominates when there's clear improvement signal
    ucb_weight = 0.1
    uncertainty_bonus = ucb_weight * sigma_safe

    # Combined acquisition before cost normalization
    raw = ei + uncertainty_bonus

    # Normalize by cost (higher cost = lower score)
    # cost is in [1, 2] per the problem (1 + mean(x_scaled))
    score = raw / np.maximum(cost, 1e-9)

    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
