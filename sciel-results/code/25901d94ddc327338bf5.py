"""Cost-aware acquisition function with smooth adaptive parameter optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE_MARGIN = 3.0  # seconds before the hard 330s limit


def _elapsed():
    return time.time() - _START_TIME


def _deadline_exceeded():
    return _elapsed() > (330.0 - _DEADLINE_MARGIN)


def _compute_ei(mu, sigma, f_best):
    """Compute Expected Improvement."""
    sigma_safe = np.maximum(sigma, 1e-9)
    improvement = f_best - mu
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.where(np.isfinite(ei), ei, 0.0)


def _compute_ucb(mu, sigma, f_best, kappa):
    """Compute Upper Confidence Bound for minimization."""
    sigma_safe = np.maximum(sigma, 1e-9)
    ucb = f_best - mu + kappa * sigma_safe
    return np.where(np.isfinite(ucb), ucb, 0.0)


def _smooth_transition(x, low, high):
    """Smooth sigmoid-like transition from 0 to 1 as x goes from low to high."""
    if high <= low:
        return 0.5
    t = np.clip((x - low) / (high - low), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization (powell 4d).

    Uses a smooth blend of Expected Improvement and UCB, both cost-normalized.
    The blend weight transitions smoothly from exploration (UCB) to exploitation
    (EI) as f_best decreases.

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
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        mu = np.where(np.isfinite(mu), mu, 1e10)
        sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
        cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
        score = (f_best - mu + 0.5 * sigma) / cost
        return np.where(np.isfinite(score), score, -1e10)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Guard against non-finite or invalid values
    mu = np.where(np.isfinite(mu), mu, 1e10)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    sigma_safe = np.maximum(sigma, 1e-9)

    # Compute EI and UCB
    ei = _compute_ei(mu, sigma, f_best)

    # Adaptive kappa for UCB: higher when f_best is large (early), lower when small (late)
    # kappa ranges from ~2.5 (explore) to ~0.5 (exploit)
    t_exploit = _smooth_transition(f_best, 0.0, 5.0)
    kappa = 0.5 + 2.0 * t_exploit
    ucb = _compute_ucb(mu, sigma, f_best, kappa)

    # Smooth blend weight: 0 = pure UCB (explore), 1 = pure EI (exploit)
    # Transition from exploration to exploitation as f_best decreases
    blend = _smooth_transition(-f_best, -5.0, 0.0)

    # Blend EI and UCB
    blended = (1.0 - blend) * ucb + blend * ei

    # Cost-normalize
    score = blended / cost

    # Ensure finiteness
    score = np.where(np.isfinite(score), score, -1e10)

    return score
# EVOLVE-BLOCK-END
