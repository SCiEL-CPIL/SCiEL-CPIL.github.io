"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: track start time to ensure we return before 330s
_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a lower-bound surrogate objective.
    
    The score is PI(f_best - epsilon) / cost, where PI is the probability
    of finding a point better than f_best - epsilon. This focuses search
    on high-confidence improvements over a fixed threshold.
    
    We blend this with a smooth LCB term to avoid step-function behavior
    while retaining exploitation focus.
    """
    # Check deadline - if close to limit, return a simple fallback
    elapsed = time.time() - _start_time
    if elapsed > 320.0:  # Safety margin before 330s kill
        cost_safe = np.maximum(cost, 1e-8)
        fallback_scores = -mu / cost_safe
        return np.where(np.isfinite(fallback_scores), fallback_scores, 0.0)
    
    # Ensure sigma is safe for division
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Epsilon for the improvement threshold
    # Use a small fixed epsilon to balance exploration and exploitation
    epsilon = 0.01
    target = f_best - epsilon
    
    # Calculate z-score for the improvement threshold
    z = (target - mu) / sigma_safe
    
    # Clip z to prevent numerical issues
    z = np.clip(z, -10, 10)
    
    # Probability of Improvement: P(f(x) < target) = Phi(z)
    prob_improve = norm.cdf(z)
    
    # Smooth LCB term: -mu - kappa * sigma
    # This provides a smooth surrogate that encourages exploration
    # kappa is chosen to balance exploitation and exploration
    kappa = 1.0
    lcb = -mu - kappa * sigma_safe
    
    # Normalize LCB to be comparable to PI
    # LCB can be negative, so we shift and scale
    lcb_min = np.min(lcb)
    lcb_max = np.max(lcb)
    lcb_range = lcb_max - lcb_min
    if lcb_range > 1e-8:
        lcb_normalized = (lcb - lcb_min) / lcb_range
    else:
        lcb_normalized = np.zeros_like(lcb)
    
    # Blend PI and LCB
    # Weight of 0.7 for PI and 0.3 for LCB
    # This emphasizes high-confidence improvements while maintaining smooth gradients
    w_pi = 0.7
    w_lcb = 0.3
    combined_score = w_pi * prob_improve + w_lcb * lcb_normalized
    
    # Normalize by cost to get cost-aware score
    scores = combined_score / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
