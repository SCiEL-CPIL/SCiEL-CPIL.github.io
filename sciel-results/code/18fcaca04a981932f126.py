"""Cost-aware acquisition for Griewank-2D BO using Thompson Sampling."""
import numpy as np
import time

# Wall-clock deadline guard (330 s kill, use 325 s safety margin)
_T0 = time.time()
_DEADLINE = 325.0


def _remaining():
    """Return seconds remaining before deadline; 0 if exceeded."""
    return max(0.0, _DEADLINE - (time.time() - _T0))


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition score using Thompson Sampling for minimizing Griewank-2D.

    Samples a single function realization f ~ N(mu, sigma) and scores candidates
    by -f(x) / cost(x), naturally coupling exploration with cost efficiency.

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

    # Clamp sigma and cost to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)

    # Thompson Sampling: draw a single sample from the posterior
    # f_sample(x) ~ N(mu(x), sigma(x))
    eps = np.random.normal(0.0, 1.0, size=mu.shape)
    f_sample = mu + sigma_safe * eps

    # Score: higher is better. We want low f and low cost.
    # Use -f_sample / cost_safe so that lower f_sample and lower cost give higher score.
    score = -f_sample / cost_safe

    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)

    return score
