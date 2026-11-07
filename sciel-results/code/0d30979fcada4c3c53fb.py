"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_start_time = time.time()
TIME_LIMIT = 330.0
SAFETY_MARGIN = 5.0

# Global evaluation counter to track progress through the optimization
_eval_count = 0


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of michalewicz 2d.
    Uses a hybrid strategy combining Lower Confidence Bound (LCB) for exploration
    and Expected Improvement (EI) for exploitation, with adaptive weighting.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the Gaussian process posterior at candidate points.
    sigma : array-like, shape (2000,)
        Standard deviation of the Gaussian process posterior at candidate points.
    f_best : float
        Best (lowest) observed function value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising to evaluate next.
    """
    global _eval_count
    _eval_count += 1
    
    # Deadline guard: return safe scores if approaching time limit
    elapsed = time.time() - _start_time
    if elapsed > TIME_LIMIT - SAFETY_MARGIN:
        # If we are at the deadline, return a neutral score to force a selection quickly
        return -mu
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Compute Lower Confidence Bound (LCB) for minimization
    # We want to maximize the score. Low mu (good mean) and High sigma (good uncertainty) should yield high score.
    # Score = -mu + beta * sigma
    # Use adaptive beta that starts high for exploration and decays for exploitation
    # Beta starts at 4.0 and decays exponentially with a time constant of 10 evaluations
    beta = 1.0 + 3.0 * np.exp(-_eval_count / 10.0)
    lcb = -mu + beta * sigma_safe
    
    # Normalize both EI and LCB to [0, 1] for blending
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min + 1e-10
    ei_normalized = (ei - ei_min) / ei_range
    
    lcb_min = np.min(lcb)
    lcb_max = np.max(lcb)
    lcb_range = lcb_max - lcb_min + 1e-10
    lcb_normalized = (lcb - lcb_min) / lcb_range
    
    # Adaptive blending: early on favor LCB (exploration), later favor EI (exploitation)
    # Use exponential decay for the LCB weight
    lcb_weight = np.exp(-_eval_count / 15.0)
    
    # Blend the two strategies
    scores = lcb_weight * lcb_normalized + (1.0 - lcb_weight) * ei_normalized
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
