"""Cost-aware acquisition function with POI-based approach for Powell 4D."""
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


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization (powell 4d).

    Uses probability of improvement (POI) combined with a deterministic regret
    lower bound, cost-normalized. The formula is:
        score = (f_best - mu + beta * sigma) / cost
    where beta is an adaptive exploration parameter.

    This directly optimizes expected reduction in simple regret per unit cost,
    avoiding the non-linear Gaussian CDF/PDF terms of EI that can be overly
    exploitative.

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

    # Adaptive beta: exploration coefficient
    # When f_best is large (early phase), use higher beta for more exploration
    # When f_best is small (late phase), use lower beta for exploitation
    # beta ranges from ~2.0 (explore) to ~0.5 (exploit)
    t_exploit = np.clip(f_best / 3.0, 0.0, 1.0)
    beta = 0.5 + 1.5 * t_exploit

    # Core score: expected improvement lower bound per unit cost
    # (f_best - mu) is the deterministic improvement
    # beta * sigma adds uncertainty-based exploration
    score = (f_best - mu + beta * sigma_safe) / cost

    # Ensure finiteness
    score = np.where(np.isfinite(score), score, -1e10)

    return score
# EVOLVE-BLOCK-END
