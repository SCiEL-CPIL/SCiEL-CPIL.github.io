"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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
    Cost-aware acquisition: Expected Improvement (EI) per unit cost,
    with a small exploration bonus to encourage diversity early on.
    """
    # Deadline guard: if we're running out of time, return a simple, fast score
    if not _deadline_ok():
        # Simple fallback: prefer low mu and low cost
        return -(mu + 0.5 * cost)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)

    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Cost-aware: EI per unit cost
    # Add a small epsilon to avoid division by zero in cost
    cost_safe = np.maximum(cost, 1e-6)

    score = ei / cost_safe

    # Small exploration bonus: add a tiny amount of UCB-like term
    # This helps escape local optima and explore unexplored regions
    # The bonus is scaled by sigma to favor high-uncertainty points
    exploration = 0.1 * sigma_safe
    score = score + exploration

    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
