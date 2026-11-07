"""Rewritten acquisition: pure UCB for minimization on the Weierstrass 1D task."""
import numpy as np
import time


def acquisition(mu, sigma, f_best):
    """
    UCB-based acquisition for minimizing the Weierstrass 1D function.

    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP posterior mean in objective units.
    sigma : np.ndarray, shape (2000,)
        GP posterior standard deviation in objective units.
    f_best : float
        Lowest observed function value so far.

    Returns
    -------
    np.ndarray, shape (2000,)
        Acquisition scores; higher values indicate candidates to evaluate next.
    """
    start = time.time()
    deadline = start + 2.0  # generous margin well under the 300 s per-eval limit

    # Sanitize inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma = np.maximum(sigma, 1e-8)
    mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e-6, neginf=1e-6)

    # Pure UCB for minimization: score = -mu + beta * sigma
    # With the Weierstrass function's many local minima and a small evaluation
    # budget (30 total), aggressive exploration helps escape poor basins.
    beta = 2.5
    scores = -mu + beta * sigma

    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -1e30)

    # Time guard
    if time.time() > deadline:
        return scores

    return scores
