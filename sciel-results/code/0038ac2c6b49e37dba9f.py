"""Cost-aware acquisition function for Bayesian optimization (4 test functions)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_T0 = time.monotonic()
_DEADLINE = 325.0  # safety margin before 330s kill


def _elapsed():
    return time.monotonic() - _T0


def _deadline_ok():
    return _elapsed() < _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware hybrid acquisition: Expected Improvement + cost-normalized exploration.
    EI handles exploitation (naturally zero when mu >= f_best), and a scaled
    sigma/cost term handles exploration in uncertain regions.
    """
    # Deadline guard: if we're running out of time, return a fast fallback
    if not _deadline_ok():
        return -(mu + 0.5 * cost)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Defensive: ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    # Avoid division by zero and negative sigma
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-6)

    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    # EI is 0 when mu >= f_best (no expected improvement), which is correct.
    diff = f_best - mu  # positive when mu < f_best (potential improvement)
    z = diff / sigma_safe
    # Clip z to avoid overflow in exp for very negative z
    z_clipped = np.clip(z, -10, 10)
    ei = diff * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    # EI should be >= 0; clip small negatives from numerical error
    ei = np.maximum(ei, 0.0)

    # Exploration term: sigma / cost rewards uncertain, cheap points
    # This is distinct from UCB/cost — it doesn't include the (f_best - mu) bias
    explore = sigma_safe / cost_safe

    # Hybrid: combine exploitation (EI) and exploration (explore)
    # Weight exploration moderately so it doesn't dominate when EI is large
    w_explore = 0.5
    score = ei / cost_safe + w_explore * explore

    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
