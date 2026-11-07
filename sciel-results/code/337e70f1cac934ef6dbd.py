"""
Acquisition function for Bayesian optimization of Styblinski-Tang 1D.
Uses a direct blend of Expected Improvement (EI) and Upper Confidence Bound (UCB).
The blend weight adapts based on the progress toward the known global minimum,
shifting from exploration (high UCB weight) to exploitation (high EI weight).
This avoids the potential instability of the Kriging Believer surrogate shift.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard
_START_TIME = time.monotonic()
_TIME_BUDGET = 300.0
_MARGIN = 10.0

# Known global minimum of Styblinski-Tang 1D
GLOBAL_MIN = -39.1662
PROGRESS_SPAN = 150.0


def _expected_improvement(mu, sigma, f_best):
    """
    Calculate Expected Improvement for minimization.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)


def _min_max_normalize(x):
    """
    Min-max normalize an array to [0, 1].
    """
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    if x_range < 1e-12:
        return np.zeros_like(x)
    return (x - x_min) / x_range


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for 2000 candidate points.
    """
    # Deadline guard
    if time.monotonic() - _START_TIME > _TIME_BUDGET - _MARGIN:
        mu = np.where(np.isfinite(mu), mu, f_best)
        return -mu

    # Ensure inputs are arrays and finite
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)

    # 1. Calculate Progress
    diff = f_best - GLOBAL_MIN
    if diff <= 0:
        progress = 1.0
    else:
        # Use a smoother transition, e.g., sigmoid-like or just linear clip
        # Linear is fine, but let's ensure it's robust
        progress = 1.0 - np.clip(diff / PROGRESS_SPAN, 0.0, 1.0)
    
    # 2. Calculate Standard EI
    ei_scores = _expected_improvement(mu, sigma, f_best)
    
    # 3. Calculate UCB Score (higher is better for exploration)
    # UCB = -(mu + beta * sigma). We want to maximize this, so we minimize mu + beta*sigma.
    # Let's use a beta that is high early and low late.
    # Early: beta ~ 2.0, Late: beta ~ 0.5
    beta = 0.5 + 1.5 * (1.0 - progress)
    ucb_scores = -(mu + beta * sigma)
    
    # 4. Normalize scores to [0, 1] to allow blending
    # Note: EI is always >= 0. UCB can be any value.
    ei_norm = _min_max_normalize(ei_scores)
    ucb_norm = _min_max_normalize(ucb_scores)
    
    # 5. Blend
    # Early (progress~0): w_explore=0.7, w_exploit=0.3
    # Late (progress~1): w_explore=0.1, w_exploit=0.9
    w_explore = 0.1 + 0.6 * (1.0 - progress)
    w_exploit = 1.0 - w_explore
    
    scores = w_exploit * ei_norm + w_explore * ucb_norm
    
    # 6. Ensure Finite Output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
