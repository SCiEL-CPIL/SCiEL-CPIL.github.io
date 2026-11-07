"""Cost-aware acquisition function combining EI with exploration bonus."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = None

def _get_start_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _start_time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    with an exploration bonus based on predictive uncertainty.
    """
    _get_start_time()
    
    # Deadline guard: track elapsed time to avoid timeout
    elapsed = time.time() - _start_time
    if elapsed > 315.0:  # 15 second safety margin before 330s kill
        # Fallback: simple EI without cost normalization
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        return np.where(np.isfinite(ei), ei, 0.0)
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Compute Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost
    ei_per_cost = ei / cost_safe
    
    # Exploration bonus: reward high uncertainty
    # Normalize sigma to [0, 1] range for the bonus
    sigma_max = np.max(sigma_safe)
    if sigma_max > 0:
        sigma_norm = sigma_safe / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # Blend EI with exploration bonus
    # Weight exploration more when EI is small (near convergence)
    ei_scale = np.max(ei_per_cost)
    if ei_scale > 0:
        ei_norm = ei_per_cost / ei_scale
    else:
        ei_norm = np.zeros_like(ei_per_cost)
    
    # Exploration weight: higher when EI is small
    exploration_weight = 0.15
    
    scores = ei_norm + exploration_weight * sigma_norm
    
    # Final safety: ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
