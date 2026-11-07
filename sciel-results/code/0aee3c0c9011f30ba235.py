"""Cost-aware acquisition for Rastrigin-2D Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_DEADLINE_MARGIN = 5.0  # seconds


def acquisition(mu, sigma, f_best, cost):
    """Return 2000 finite scores (higher = evaluate next).

    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
    sigma : np.ndarray, shape (2000,)
    f_best : float
    cost : np.ndarray, shape (2000,)
    """
    t0 = time.time()

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)

    n = len(mu)

    # Safety: ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    sigma = np.maximum(sigma, 1e-9)

    # Compute standard EI/cost
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    base_score = ei / cost

    # Add an exploration bonus that rewards uncertainty
    # UCB-style: f_best - mu + beta * sigma, but we want "improvement" direction
    # Since lower is better, exploration = sigma (uncertainty in f)
    beta = 0.5
    ucb_term = (f_best - mu + beta * sigma) / cost

    # Blend: use max of the two to ensure we always have a positive signal
    # when EI is zero but there's still uncertainty
    score = np.maximum(base_score, ucb_term * 0.3)

    # Deadline guard
    if time.time() - t0 > 2.0:
        # Fallback: simple 1/cost weighted by (f_best - mu) + sigma
        fallback = (f_best - mu + sigma) / cost
        fallback = np.where(np.isfinite(fallback), fallback, 0.0)
        return fallback

    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    return score
# EVOLVE-BLOCK-END
