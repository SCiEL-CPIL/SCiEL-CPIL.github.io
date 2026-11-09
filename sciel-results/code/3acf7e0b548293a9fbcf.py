"""Cost-aware acquisition function for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = 330.0
_SAFETY_MARGIN = 5.0


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME) - _SAFETY_MARGIN


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid of EI and UCB,
    with adaptive exploration based on uncertainty.
    """
    if _time_left() <= 0:
        return -mu / (cost + 1e-9)

    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Expected Improvement for minimization
    improvement = f_best - mu
    z = improvement / sigma_safe
    z = np.clip(z, -50, 50)
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # UCB (Upper Confidence Bound) for minimization: lower is better,
    # so we want low mu + high uncertainty. For scoring (higher = better),
    # use -mu + kappa * sigma
    # Adaptive kappa: increase exploration early, decrease as we converge
    # Use a simple heuristic: kappa based on the ratio of current best to initial scale
    # Since we don't know initial scale, use a fixed but moderate kappa
    # kappa = 1.0 is a common choice; let's try 0.5 to be more exploitative
    kappa = 0.5
    ucb_score = -mu + kappa * sigma_safe

    # Normalize both to [0, 1]
    ei_max = np.max(ei)
    ei_min = np.min(ei)
    if ei_max > ei_min:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.zeros_like(ei)

    ucb_max = np.max(ucb_score)
    ucb_min = np.min(ucb_score)
    if ucb_max > ucb_min:
        ucb_norm = (ucb_score - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros_like(ucb_score)

    # Blend: 60% EI, 40% UCB
    alpha = 0.6
    blended = alpha * ei_norm + (1 - alpha) * ucb_norm

    # Cost-aware normalization
    scores = blended / cost_safe
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
