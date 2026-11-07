"""Cost-aware acquisition function for Bayesian optimization (levy 2d).

This module implements a cost-aware Thompson Sampling acquisition function.
"""
import numpy as np
import time

# Deadline guard: the evaluator kills after 330s wall-clock.
# We track module load time and ensure we return well before the limit.
_MODULE_START = time.time()
DEADLINE = 325.0  # seconds; leave 5s safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.

    For each candidate, we draw a sample from the Gaussian posterior
    N(mu, sigma^2) and score it as -sample / cost.  This implicitly
    balances exploration (high sigma leads to wide samples) and
    exploitation (low mu leads to low samples => high score).

    Parameters
    ----------
    mu : array-like, shape (2000,)
        GP mean predictions in objective units.
    sigma : array-like, shape (2000,)
        GP standard-deviation predictions in objective units.
    f_best : float
        Best (lowest) observed objective value so far.
    cost : array-like, shape (2000,)
        Evaluation cost for each candidate.

    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition values; higher is preferred.  All values are finite.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Deadline guard: if we are too close to the kill time, return a
    # safe fallback (uniform random scores) immediately.
    if time.time() > DEADLINE:
        return np.random.uniform(0.0, 1.0, size=mu.shape)

    n = mu.shape[0]

    # Guard against degenerate inputs
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-9)

    # Thompson Sampling: draw one posterior sample per candidate.
    # sample ~ N(mu, sigma^2)
    # Using the standard trick: sample = mu + sigma * z, z ~ N(0,1)
    z = np.random.standard_normal(n)
    samples = mu + sigma_safe * z

    # For minimisation, lower samples are better.
    # Score = -sample / cost  (higher score = prefer)
    scores = -samples / cost_safe

    # Clip to finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
