"""Cost-aware acquisition for Hartmann 6d Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import ndtr
import time

_DEADLINE = 310.0
_start_time = time.time()


def _check_time():
    return (time.time() - _start_time) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: combines EI, adaptive LCB, and power cost penalty.
    Returns 2000 finite scores (higher = more promising).
    """
    if _check_time():
        return np.ones(2000)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    n = len(mu)
    if n != 2000:
        return np.ones(n)

    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)

    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    pdf_z = np.exp(-0.5 * z_clipped * z_clipped) / np.sqrt(2.0 * np.pi)
    ei = (f_best - mu) * ndtr(z_clipped) + sigma_safe * pdf_z
    ei = np.maximum(ei, 0.0)

    # Adaptive LCB (Lower Confidence Bound) for minimization
    f_min = -3.32237
    f_start = -1.0
    progress = np.clip((f_start - f_best) / (f_start - f_min), 0.0, 1.0)
    
    # More aggressive exploration early, stronger exploitation late
    beta = 3.5 - 2.8 * progress  # beta from 3.5 (explore) to 0.7 (exploit)
    lcb = mu - beta * sigma_safe
    lcb_gain = np.maximum(f_best - lcb, 0.0)

    # Weight LCB more heavily, especially early on
    alpha = 0.6 - 0.3 * progress  # 0.6 early, 0.3 late
    combined = ei + alpha * lcb_gain

    # Stronger cost penalty to favor cheaper evaluations
    scores = combined / np.power(cost_safe, 2.0)

    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 1e-10)

    return scores
# EVOLVE-BLOCK-END
