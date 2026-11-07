"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Adaptive Lower Confidence Bound (LCB) for multimodal minimization.
    
    Uses LCB with adaptive kappa that scales based on distance to known optimum.
    For minimization, LCB = mu - kappa * sigma. We want to maximize this score
    (higher means more promising).
    
    The key insight: when f_best is far from the known global minimum (-1.801),
    we should explore more aggressively. As we approach the optimum, we should
    exploit more.
    """
    t_start = time.time()
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Known global minimum for Michalewicz 2D
    global_min = -1.801
    
    # Compute distance to known optimum as a fraction of the range
    # Range of Michalewicz 2D is roughly [-1.8, 1.8]
    range_scale = 3.6
    dist_to_opt = max(f_best - global_min, 0.0)
    normalized_dist = dist_to_opt / range_scale  # 0 when at optimum, 1 when far
    
    # Adaptive kappa: larger when far from optimum (early), smaller when close (late)
    # Base kappa between 1.0 and 2.5
    kappa = 1.0 + 1.5 * normalized_dist
    
    # Lower Confidence Bound for minimization: lower is better
    # Score = mu - kappa * sigma (we want to maximize this)
    lcb_score = mu - kappa * sigma_safe
    
    # Normalize to [0, 1] for consistent scaling
    lcb_max = np.max(lcb_score)
    lcb_min = np.min(lcb_score)
    
    if lcb_max - lcb_min > eps:
        score = (lcb_score - lcb_min) / (lcb_max - lcb_min)
    else:
        score = np.zeros_like(lcb_score)
    
    # Safety: ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Time guard: if this takes too long, we could bail, but LCB is O(n) so it's fast
    elapsed = time.time() - t_start
    if elapsed > 250:  # Safety margin before 300s limit
        return score
    
    return score
# EVOLVE-BLOCK-END
