"""Cost-aware acquisition for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE_MARGIN_S = 5.0
_START_TIME = time.time()


def _elapsed():
    return time.time() - _START_TIME


def _deadline_reached():
    return _elapsed() > (330.0 - _DEADLINE_MARGIN_S)


def acquisition(mu, sigma, f_best, cost):
    if _deadline_reached():
        idx = np.argmin(cost)
        scores = np.full(2000, -1e30, dtype=np.float64)
        scores[idx] = 0.0
        return scores

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)

    # Structural change: Thompson Sampling-based cost-aware acquisition.
    # Sample from the GP posterior and score by optimistic improvement per unit cost.
    # Use a small number of samples to reduce variance while retaining stochastic exploration.
    n_samples = 8
    rng = np.random.default_rng()

    # Shape: (n_samples, 2000)
    samples = mu[None, :] + sigma[None, :] * rng.standard_normal((n_samples, mu.size))

    # For each sample, compute (f_best - sample) / cost
    # Shape: (n_samples, 2000)
    scores_per_sample = (f_best - samples) / cost[None, :]

    # Use the mean over samples for a stable score
    score = np.mean(scores_per_sample, axis=0)

    # Ensure finite
    score = np.where(np.isfinite(score), score, -1e30)
    score = np.clip(score, -1e10, 1e10)

    return score
# EVOLVE-BLOCK-END
