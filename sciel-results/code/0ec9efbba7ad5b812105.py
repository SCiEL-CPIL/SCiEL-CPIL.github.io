"""Rewritten program: cost-aware acquisition for BO on shifted Hartmann-3D."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Module-level deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 295.0  # safety margin before 300s / 330s kill


def _time_ok():
    return time.time() < _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.

    Parameters
    ----------
    mu : (N,) array, GP mean in objective units
    sigma : (N,) array, GP std in objective units
    f_best : float, best observed value (lowest)
    cost : (N,) array, cost of evaluating each candidate

    Returns
    -------
    (N,) array of finite scores; higher = evaluate next.
    """
    n = len(mu)
    if not _time_ok():
        # Fallback: just rank by low mu / high uncertainty
        return np.where(np.isfinite(mu), -mu, -1e6)

    sigma_safe = np.maximum(sigma, 1e-8)

    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    ei = (f_best - mu) * cdf + sigma_safe * pdf

    # Guard against non-finite
    ei = np.where(np.isfinite(ei), ei, 0.0)

    # Cost-aware: divide by cost (higher cost penalizes)
    cost_safe = np.maximum(cost, 1e-8)
    score = ei / cost_safe

    # Additional structural tweak: penalize very low sigma (exploit too hard)
    # Add a small UCB-like term to encourage exploration in uncertain regions
    # This makes it a hybrid: EI/cost + small exploration bonus
    exploration_bonus = 0.1 * sigma_safe / cost_safe

    score = score + exploration_bonus

    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
