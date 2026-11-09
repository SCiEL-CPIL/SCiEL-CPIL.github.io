"""Cost-aware acquisition function for Bayesian optimization (styblinski tang 2d)."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-normalized Lower Confidence Bound (LCB) acquisition function.

    Parameters
 ----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        Best (lowest) observed function value.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.

    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher is better).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    sigma = np.clip(sigma, 1e-8, None)
    cost = np.clip(cost, 1e-8, None)

    kappa = 1.1

    scores = (kappa * sigma - mu) / cost

    scores = np.where(np.isfinite(scores), scores, -1e18)

    return scores
# EVOLVE-BLOCK-END
