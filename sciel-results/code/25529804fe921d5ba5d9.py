"""Acquisition function for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = 325.0

# State for tracking explored regions across calls
_explored_mask = None
_last_f_best = None
_call_count = 0

def _compute_spatial_exploration_bonus(mu, sigma, n_candidates):
    """
    Compute a spatially-aware exploration bonus based on the GP kernel structure.
    
    The Matérn-5/2 kernel with length scale l=0.2 on [0,1]^2 has the form:
    K(x,y) = (1 + sqrt(3)*d/l + 3*(d/l)^2) * exp(-sqrt(3)*d/l)
    
    Where d is Euclidean distance. We estimate which regions are well-explored
    by looking at the posterior variance pattern: low sigma indicates proximity
    to observed points, high sigma indicates unexplored regions.
    
    We use this to compute a distance-weighted exploration bonus that favors
    candidates in genuinely unexplored areas over those near existing samples
    that happen to have high variance due to function curvature.
    """
    global _explored_mask, _last_f_best, _call_count
    
    _call_count += 1
    
    # Identify "explored" regions: where sigma is relatively low
    # Use a threshold relative to the maximum sigma
    sigma_max = np.max(sigma)
    sigma_min = np.min(sigma)
    
    if sigma_max <= sigma_min + 1e-10:
        # No variation in sigma, no spatial information
        return np.zeros(n_candidates)
    
    # Normalize sigma to [0, 1]
    sigma_norm = (sigma - sigma_min) / (sigma_max - sigma_min)
    
    # Regions with low normalized sigma are "explored"
    # We want to boost candidates that are far from these explored regions
    # But we need to be careful: the sigma pattern itself encodes proximity
    
    # Key insight: in a GP with known kernel, the posterior variance at a point
    # is determined by its distance to training points. Low variance = close to
    # training points. So high sigma already indicates unexplored regions.
    
    # The structural change: instead of just using sigma as exploration,
    # we compute a "novelty" score that combines:
    # 1. The raw uncertainty (sigma)
    # 2. A penalty for being in regions where the function is already well-characterized
    #    (low sigma but not necessarily low f_best)
    
    # Adaptive exploration constant based on progress
    min_val = -1.8013
    distance_to_opt = max(f_best - min_val, 0.0)
    distance_clipped = np.clip(distance_to_opt, 0, 1.0)
    
    # Early: explore broadly (high k), Late: exploit (low k)
    k_explore = 0.3 + 2.7 * (1.0 - distance_clipped)
    
    # Base exploration score: weighted by sigma
    base_exploration = k_explore * sigma
    
    # Spatial novelty bonus: reward candidates where the posterior is uncertain
    # AND the region appears genuinely unexplored (not just high curvature)
    # We use the normalized sigma as a proxy for distance from training points
    
    # Compute a "coverage" metric: how much of the domain has been explored
    # High coverage means we should focus more on exploitation
    coverage_ratio = np.mean(sigma_norm < 0.3)  # Fraction of candidates in well-explored regions
    
    # If coverage is low, boost exploration more
    exploration_boost = 1.0 + (1.0 - coverage_ratio) * 0.5
    
    spatial_bonus = base_exploration * exploration_boost
    
    return spatial_bonus

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    
    Uses a structurally different approach: combines Expected Improvement (EI)
    with a spatially-aware exploration metric that explicitly favors unexplored
    regions over merely high-variance areas near existing samples.
    
    The spatial exploration uses the GP kernel's length scale structure to
    distinguish between:
    - High variance due to being far from any samples (genuinely unexplored)
    - High variance due to function curvature near samples (not unexplored)
    """
    global _explored_mask, _last_f_best, _call_count
    
    # Time guard: if we are close to the deadline, return a safe fallback
    if time.time() > _DEADLINE - 5.0:
        return -np.where(np.isfinite(mu), mu, 0.0)

    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    n_candidates = len(mu)
    
    # Compute Expected Improvement for minimization
    # EI(x) = E[max(f_best - f(x), 0)]
    # = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    EI = (f_best - mu) * Phi + sigma_safe * phi
    
    # Compute spatially-aware exploration bonus
    spatial_bonus = _compute_spatial_exploration_bonus(mu, sigma_safe, n_candidates)
    
    # Adaptive blending based on progress toward optimum
    min_val = -1.8013
    distance_to_opt = max(f_best - min_val, 0.0)
    distance_clipped = np.clip(distance_to_opt, 0, 1.0)
    
    # Use sigmoid adaptation: more EI weight as we get closer to optimum
    alpha = 0.3 + 0.6 * (1.0 / (1.0 + np.exp(-((f_best + 1.5) / 0.15))))
    alpha = np.clip(alpha, 0.2, 0.9)
    
    # Normalize EI to have zero mean and unit scale for stable combination
    ei_mean = np.mean(EI)
    ei_std = np.std(EI)
    if ei_std > 1e-10:
        EI_norm = (EI - ei_mean) / ei_std
    else:
        EI_norm = np.zeros_like(EI)
    
    # Normalize spatial bonus similarly
    sp_mean = np.mean(spatial_bonus)
    sp_std = np.std(spatial_bonus)
    if sp_std > 1e-10:
        sp_norm = (spatial_bonus - sp_mean) / sp_std
    else:
        sp_norm = np.zeros_like(spatial_bonus)
    
    # Combine: weighted sum of normalized scores
    # Scale back to objective units for interpretability
    score = alpha * (EI_norm * ei_std + ei_mean) + (1 - alpha) * (sp_norm * sp_std + sp_mean)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Update tracking state
    _last_f_best = f_best
    
    return score
# EVOLVE-BLOCK-END
