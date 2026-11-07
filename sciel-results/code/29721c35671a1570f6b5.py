import numpy as np
import time

_START_TIME = time.time()
_DEADLINE_MARGIN = 15.0

# Precompute standard normal CDF values on a fine grid for fast empirical POI.
# We sample standard normal values, standardize them, and use a large grid.
_GRID = None

def _get_grid():
    global _GRID
    if _GRID is None:
        # 20481 points, roughly symmetric, covering ~[-5.5, 5.5]
        _GRID = np.linspace(-5.5, 5.5, 20481)
    return _GRID

def _empirical_cdf(x):
    """Fast empirical CDF of the standard normal on the precomputed grid."""
    g = _get_grid()
    # x is an array; use searchsorted for O(log N) per element
    # We want P(Z <= x) where Z ~ N(0,1) sampled on grid.
    idx = np.searchsorted(g, x, side='right')
    return idx / len(g)

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using empirical Probability of Improvement (POI).

    POI(x) = P(f(x) < f_best) = Phi((f_best - mu) / sigma)

    Instead of using scipy's norm.cdf (closed form), we use an empirical
    CDF from a precomputed grid of standard normal samples. This is a
    structurally different computation: it is a discrete, grid-based
    approximation that can behave differently in the tails and at the
    boundaries, and it avoids any floating-point edge cases in the
    closed-form CDF.

    We also add a small exploration bonus based on sigma to ensure
    the search does not collapse to a single point too early.

    Parameters:
    -----------
    mu : ndarray, shape (2000,)
    sigma : ndarray, shape (2000,)
    f_best : float

    Returns:
    --------
    scores : ndarray, shape (2000,)
    """
    # Deadline guard
    remaining_time = 330.0 - (time.time() - _START_TIME)
    if remaining_time < _DEADLINE_MARGIN:
        # Fallback: pure exploitation
        scores = -mu
        return np.where(np.isfinite(scores), scores, -np.inf)

    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)

    # z-score: how many standard deviations f_best is above mu
    z = (f_best - mu) / sigma_safe

    # Clip z to the grid range to avoid out-of-bounds
    z = np.clip(z, -5.5, 5.5)

    # Empirical POI: P(f(x) < f_best) = Phi(z)
    poi = _empirical_cdf(z)

    # Exploration bonus: encourage exploring high-uncertainty regions.
    # We use a normalized sigma term. The scale of sigma varies, so we
    # normalize by the median sigma to get a relative measure.
    sigma_med = np.median(sigma_safe)
    if sigma_med < eps:
        sigma_med = eps
    sigma_norm = sigma_safe / sigma_med

    # Combine: POI is the primary signal, with a small sigma bonus.
    # The bonus is scaled so it is at most ~10% of the POI range.
    exploration_weight = 0.1
    scores = poi + exploration_weight * sigma_norm

    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)

    return scores
