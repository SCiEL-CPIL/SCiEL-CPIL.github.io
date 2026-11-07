import numpy as np
from scipy.stats import norm
import time

_START_TIME = None
_DEADLINE_SAFETY = 5.0

def acquisition(mu, sigma, f_best, cost):
    """
    Diversity-aware cost-normalized Expected Improvement.
    
    Strategy:
    1. Compute standard Expected Improvement (EI).
    2. Normalize by evaluation cost (cost-aware).
    3. Compute a 'crowding' penalty based on local GP confidence.
       Regions with low sigma are considered 'well-explored' or 'dense'.
       We penalize candidates that fall into these dense, low-uncertainty regions
       to encourage the search to move to unexplored areas (high sigma) or
       areas with high potential improvement but less redundancy.
    4. The penalty is scaled to be significant enough to break local optima
       in the Levy function's many local minima.
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    elapsed = time.time() - _START_TIME
    
    # Safety guard: if we are close to the deadline, return a simple, fast score
    if elapsed > 325.0:
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        cost_safe = np.maximum(cost, 1e-9)
        score = ei / cost_safe
        score = np.where(np.isfinite(score), score, 0.0)
        return score

    # 1. Compute Expected Improvement (EI)
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Avoid numerical issues in norm.cdf/pdf for extreme z
    z_clipped = np.clip(z, -10, 10)
    
    phi_z = norm.pdf(z_clipped)
    Phi_z = norm.cdf(z_clipped)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    ei = np.maximum(ei, 0.0)
    
    # 2. Cost-normalize
    cost_safe = np.maximum(cost, 1e-9)
    base_score = ei / cost_safe
    
    # 3. Diversity/Crowding Penalty
    # The idea: If sigma is very low, the GP is confident. 
    # If we keep picking points where sigma is low and mu is not significantly better than f_best,
    # we are just refining a local area. 
    # However, simple EI already handles the "mu better than f_best" part.
    # The problem is when EI is small but positive, and we keep picking nearby points.
    
    # Let's define a "confidence" metric. High confidence = Low sigma.
    # We want to penalize points that are in high-confidence regions UNLESS they have very high EI.
    # Alternatively, we can use a "local density" proxy.
    # Since we don't have the coordinates of past points, we use the GP's sigma field.
    # Low sigma implies proximity to observed points (in the kernel sense).
    
    # Normalize sigma to [0, 1] for scaling
    sigma_min = np.min(sigma_safe)
    sigma_max = np.max(sigma_safe)
    if sigma_max > sigma_min:
        sigma_norm = (sigma_safe - sigma_min) / (sigma_max - sigma_min)
    else:
        sigma_norm = np.ones_like(sigma_safe)
    
    # Penalty is high where sigma_norm is low (dense/explored)
    # We use a power function to make the penalty sharper in very dense areas
    # penalty = (1 - sigma_norm)^2 * base_penalty_scale
    
    # We need to balance this with the base score.
    # If base_score is very high, we might ignore the penalty.
    # Let's try a multiplicative penalty or a subtractive one scaled by the base score's magnitude.
    
    # A common approach: score = EI/cost - lambda * (1 - sigma_norm) * (EI/cost)
    # This reduces the effective score in dense areas.
    
    # Let's try a subtractive penalty that is significant relative to the typical EI.
    # Typical EI values can vary. Let's scale the penalty by the max base score to be adaptive.
    max_base = np.max(base_score)
    if max_base > 1e-9:
        # Scale penalty to be a fraction of the max possible score in dense regions
        # We want to penalize dense regions by up to ~50% of the max score
        penalty = 0.5 * max_base * (1.0 - sigma_norm) ** 2
    else:
        penalty = np.zeros_like(base_score)
    
    # 4. Combine
    score = base_score - penalty
    
    # Ensure finite and non-negative
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    return score
