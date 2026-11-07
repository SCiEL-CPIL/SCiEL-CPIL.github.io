"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_DEADLINE = None
_DEADLINE_LIMIT = 310.0  # Safety margin before 330s kill

def _init_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + _DEADLINE_LIMIT

def _check_deadline():
    global _DEADLINE
    if _DEADLINE is not None and time.time() > _DEADLINE:
        return True
    return False

def acquisition(mu, sigma, f_best, cost):
    _init_deadline()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive and cost is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-6)
    
    # Compute EI for minimization
    diff = f_best - mu
    z = diff / sigma_safe
    z = np.clip(z, -10, 10)
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    ei = np.maximum(diff * cdf + sigma_safe * pdf, 0.0)
    
    # Compute LCB for minimization
    kappa = 2.0
    lcb = mu - kappa * sigma_safe
    lcb_score = -lcb  # Higher is better
    
    # Adaptive blending: weight LCB more when sigma is large relative to EI potential
    # Use the ratio of sigma to the scale of EI
    ei_scale = np.max(ei) + 1e-12
    sigma_ratio = sigma_safe / (ei_scale + sigma_safe)  # 0 to 1, higher when sigma dominates
    
    # Blend weight: more LCB when uncertainty is high relative to improvement
    # alpha = 0.5 + 0.5 * sigma_ratio, so alpha in [0.5, 1.0]
    alpha = 0.5 + 0.5 * sigma_ratio
    
    # Normalize both components
    lcb_min = np.min(lcb_score)
    lcb_max = np.max(lcb_score)
    lcb_range = lcb_max - lcb_min + 1e-12
    lcb_norm = (lcb_score - lcb_min) / lcb_range
    
    ei_norm = ei / ei_scale
    
    # Combine with adaptive weight
    combined = alpha * lcb_norm + (1 - alpha) * ei_norm
    
    # Cost-aware: divide by cost
    scores = combined / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    if _check_deadline():
        return scores
    
    return scores
# EVOLVE-BLOCK-END
