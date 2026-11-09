"""Rewritten cost-aware acquisition for Bayesian optimization of Levy 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Expected improvement per unit cost, with exponential cost penalty
    # to strongly suppress expensive (boundary) regions.
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # Exponential cost penalty: exp(-alpha * cost) where cost in [1, 2]
    # alpha=1.5 gives strong suppression of high-cost regions
    cost_penalty = np.exp(-1.5 * cost)
    scores = ei * cost_penalty
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores
# EVOLVE-BLOCK-END
