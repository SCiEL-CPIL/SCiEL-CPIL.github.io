"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard: 330s wall-clock limit, use 325s as safety margin
_DEADLINE = time.time() + 325.0


def _check_deadline():
    """Return True if we should stop to avoid timeout."""
    return time.time() >= _DEADLINE


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Branin 2D.
    
    Uses a hybrid approach:
    - Expected Improvement (EI) for exploitation
    - Upper Confidence Bound (UCB) for exploration
    - Weighted combination with adaptive exploration
    
    mu: array of shape (2000,), posterior mean
    sigma: array of shape (2000,), posterior std
    f_best: best (lowest) observed value so far
    
    Returns: array of shape (2000,) with scores (higher = more promising)
    """
    # Clamp sigma to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z_ei = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma * norm.pdf(z_ei)
    
    # Upper Confidence Bound (for minimization, we use lower confidence bound)
    # LCB = mu - kappa * sigma
    # We want to maximize the negative LCB, so score = -LCB = -mu + kappa*sigma
    # But we need to combine with EI. Let's use a weighted combination.
    
    # Adaptive exploration: increase exploration when we've found a good value
    # relative to the function range
    branin_range = 302.0  # approximate range of Branin function
    progress = max(0.0, min(1.0, (302.0 - f_best) / 302.0))
    
    # Start with more exploration, decrease as we converge
    kappa = 1.0 + 2.0 * (1.0 - progress)
    
    # LCB score (higher is better for exploration)
    lcb_score = -mu + kappa * sigma
    
    # Normalize both scores to have similar scales
    ei_max = np.max(ei)
    lcb_max = np.max(lcb_score)
    
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    if lcb_max > 1e-12:
        lcb_norm = lcb_score / lcb_max
    else:
        lcb_norm = np.zeros_like(lcb_score)
    
    # Weighted combination: blend EI and LCB
    # More weight on EI as we converge
    w_ei = 0.5 + 0.5 * progress
    w_lcb = 1.0 - w_ei
    
    score = w_ei * ei_norm + w_lcb * lcb_norm
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
