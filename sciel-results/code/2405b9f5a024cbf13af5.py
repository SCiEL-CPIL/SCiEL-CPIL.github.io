"""Acquisition function for Bayesian optimization of a fixed seeded random RBF mixture."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


# Global deadline guard
_DEADLINE = 330.0  # wall-clock seconds
_START_TIME = None


def acquisition(mu, sigma, f_best):
    """
    Acquisition function that selects a point to evaluate next.
    
    Uses a combination of Expected Improvement (EI) and Upper Confidence Bound (UCB)
    with adaptive exploration that decays over time. The function returns scores for
    2000 candidate points; higher scores mean more promising.
    
    mu: array of shape (2000,) - GP posterior mean in objective units
    sigma: array of shape (2000,) - GP posterior std dev in objective units
    f_best: float - lowest observed value so far
    
    Returns: array of shape (2000,) - scores, higher is better
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    elapsed = time.time() - _START_TIME
    remaining = _DEADLINE - elapsed - 2.0  # safety margin
    
    if remaining <= 0:
        # Emergency: return simple EI
        z = (f_best - mu) / (sigma + 1e-9)
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Infer optimization progress from the relationship between mu/sigma and f_best
    # Early on: high uncertainty, should explore more
    # Late: low uncertainty, should exploit more
    
    # Use the fraction of points where sigma is small relative to the spread of mu
    # as a proxy for how much we've learned
    sigma_mean = np.mean(sigma)
    sigma_max = np.max(sigma)
    
    # If sigma is very small everywhere, we're in late optimization -> exploit
    # If sigma is large, we're in early optimization -> explore
    
    # Adaptive beta: starts high (exploration), decays as uncertainty decreases
    # beta controls the UCB exploration term
    # We estimate progress by how concentrated sigma is
    if sigma_max > 1e-8:
        # Ratio of mean to max sigma: if close to 1, uncertainty is uniform (early)
        # If much less than 1, uncertainty is concentrated (late, some areas explored)
        progress = 1.0 - (sigma_mean / (sigma_max + 1e-12))
        # Clamp to [0, 1]
        progress = np.clip(progress, 0.0, 1.0)
    else:
        progress = 1.0
    
    # Beta decays from ~2.5 (heavy exploration) to ~0.5 (heavy exploitation)
    beta = 2.5 - 2.0 * progress
    
    # Compute EI
    z_ei = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma * norm.pdf(z_ei)
    
    # Compute UCB
    # UCB = -mu + beta * sigma (we're minimizing, so higher UCB score = more promising)
    ucb = -mu + beta * sigma
    
    # Normalize EI and UCB to comparable scales
    ei_max = np.max(ei)
    ucb_max = np.max(ucb)
    
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = ei
    
    if ucb_max > 1e-12:
        ucb_norm = ucb / ucb_max
    else:
        ucb_norm = ucb
    
    # Blend: early on weight UCB more, late on weight EI more
    # Weight for EI increases with progress
    w_ei = 0.5 + 0.5 * progress  # from 0.5 to 1.0
    w_ucb = 1.0 - w_ei * 0.5     # from 0.75 to 0.5
    
    # Actually, let's use a cleaner blend:
    # score = w_ei * EI_norm + w_ucb * UCB_norm, with w_ei + w_ucb = 1
    w_ei = 0.4 + 0.6 * progress  # 0.4 to 1.0
    w_ucb = 1.0 - w_ei
    
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
