"""Rewritten program with a UCB-based, cost-aware acquisition function."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0  # seconds before the hard 330s limit


def _elapsed():
    return time.time() - _START_TIME


def _deadline_exceeded():
    return _elapsed() > (330.0 - _DEADLINE_MARGIN)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization (powell 4d).
    Uses Upper Confidence Bound (UCB) normalized by cost.

    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Lowest observed function value so far.
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate, shape (2000,).

    Returns
    -------
    np.ndarray, shape (2000,)
        Finite scores; higher = prefer to evaluate next.
    """
    # Explicit deadline guard
    if _deadline_exceeded():
        # Return a safe, finite fallback: favor low mu / high sigma
        sigma_safe = np.maximum(sigma, 1e-9)
        return -mu / (sigma_safe + 1e-9) + np.log(sigma_safe + 1.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Guard against non-finite or invalid values
    mu = np.where(np.isfinite(mu), mu, 1e10)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    # Clamp sigma to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)

    # UCB for minimization:
    # We want to find points where the lower bound of the confidence interval is low.
    # Lower bound = mu - beta * sigma
    # We want to MINIMIZE (mu - beta * sigma), which is equivalent to MAXIMIZING (-mu + beta * sigma)
    # beta controls exploration. 2.0 is a strong exploration factor suitable for global optimization.
    beta = 2.0
    ucb_score = -mu + beta * sigma_safe

    # Cost-aware: Divide by cost to prefer cheaper evaluations with similar potential
    # Cost is in [1, 2], so this is a mild regularization.
    score = ucb_score / cost

    # Ensure finiteness
    score = np.where(np.isfinite(score), score, -1e10)

    return score
# EVOLVE-BLOCK-END
