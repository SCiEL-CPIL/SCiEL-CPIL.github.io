"""Rewritten acquisition for Bayesian optimization of ackley 2d."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_DEADLINE = time.time() + 300.0 - 5.0  # safety margin before 300 s limit


def _ei(mu, sigma, f_best):
    z = (f_best - mu) / (sigma + 1e-9)
    return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)


def _ucb(mu, sigma, f_best, xi=0.1):
    return -mu + xi * sigma


def acquisition(mu, sigma, f_best):
    # Explicit deadline guard: if we are close to the wall-clock limit,
    # fall back to a fast, finite, deterministic score.
    if time.time() > _DEADLINE:
        # Exploit lowest mean, guaranteeing finite output.
        return -mu.copy()

    sigma = np.maximum(sigma, 1e-9)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    ei = _ei(mu, sigma, f_best)
    ucb = _ucb(mu, sigma, f_best, xi=0.1)

    # Normalize each term to [0, 1] so the blend is scale-free and finite.
    def _norm(a):
        lo = np.min(a)
        hi = np.max(a)
        rng = hi - lo
        if not np.isfinite(rng) or rng <= 1e-12:
            return np.zeros_like(a)
        return (a - lo) / rng

    ei_n = _norm(ei)
    ucb_n = _norm(ucb)

    # Blend: primarily EI (exploitation) with a UCB exploration component.
    scores = 0.85 * ei_n + 0.15 * ucb_n

    # Ensure strictly finite output.
    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores
# EVOLVE-BLOCK-END
