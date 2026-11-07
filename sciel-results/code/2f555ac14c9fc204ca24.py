"""Acquisition for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_DEADLINE = time.time() + 290.0  # safety margin before 300s kill


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: blend of Expected Improvement and a
    UCB-style uncertainty bonus, with a small exploitation term.
    Higher score => evaluate next.
    """
    t0 = time.time()
    if t0 > _DEADLINE:
        # Fallback: return EI quickly
        sigma_safe = np.maximum(sigma, 1e-12)
        z = (f_best - mu) / sigma_safe
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    sigma_safe = np.maximum(sigma, 1e-12)

    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # UCB-style: mean + beta * std (for minimization, lower mean is better,
    # higher std is exploratory)
    # We want to maximize the score, so:
    #   score = -mu + beta * sigma  (exploit low mu, explore high sigma)
    ucb = -mu + 1.1 * sigma_safe

    # Normalize both to [0,1]-ish for blending
    ei_min, ei_max = ei.min(), ei.max()
    ucb_min, ucb_max = ucb.min(), ucb.max()

    ei_range = ei_max - ei_min
    ucb_range = ucb_max - ucb_min

    if ei_range < 1e-12:
        ei_norm = np.zeros_like(ei)
    else:
        ei_norm = (ei - ei_min) / ei_range

    if ucb_range < 1e-12:
        ucb_norm = np.zeros_like(ucb)
    else:
        ucb_norm = (ucb - ucb_min) / ucb_range

    # Blend: 60% EI, 40% UCB
    score = 0.6 * ei_norm + 0.4 * ucb_norm

    # Add a tiny exploitation nudge: prefer lower mu directly
    mu_range = mu.max() - mu.min()
    if mu_range > 1e-12:
        score += 0.1 * (1.0 - (mu - mu.min()) / mu_range)

    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
