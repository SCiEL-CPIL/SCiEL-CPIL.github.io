"""Cost-aware acquisition for BO on three hump camel 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_TIME_LIMIT = 320.0  # Safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement with adaptive
    multiplicative cost penalty.
    
    Strategy:
    1. Compute Expected Improvement (EI) for minimization.
    2. Apply a multiplicative cost penalty: score = EI / (cost ** gamma).
       - Gamma is adaptive: lower (softer) penalty early in the search when 
         f_best is large, higher (stronger) penalty late when f_best is small 
         to conserve budget for fine-tuning.
    3. Add a small exploration bonus in high-uncertainty regions, also 
         cost-aware, to prevent ignoring cheap exploratory points.
    
    Parameters:
        mu: (2000,) GP mean predictions in objective units
        sigma: (2000,) GP std dev predictions (positive) in objective units
        f_best: float, best observed value so far
        cost: (2000,) evaluation cost per candidate
    
    Returns:
        (2000,) finite acquisition scores, higher = better
    """
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        # Emergency: return something finite immediately
        return np.zeros(len(mu), dtype=np.float64)
    
    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1.0) # Cost is always >= 1
    
    # Standardized improvement: Z = (f_best - mu) / sigma
    # For minimization, improvement is when mu < f_best
    Z = (f_best - mu) / sigma_safe
    Z = np.clip(Z, -10.0, 10.0)  # Numerical stability
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    phi = norm.pdf(Z)
    Phi = norm.cdf(Z)
    
    EI = (f_best - mu) * Phi + sigma_safe * phi
    EI = np.maximum(EI, 0.0)
    
    # Adaptive cost penalty exponent gamma
    # f_best scale: Three hump camel 2D max is ~5.5, min is 0.
    # progress = 1 - f_best / 5.5. 
    # Early (f_best ~ 5.5): progress ~ 0 -> gamma ~ 0.5 (soft penalty)
    # Late (f_best ~ 0): progress ~ 1 -> gamma ~ 1.0 (linear penalty)
    f_scale = 5.5
    progress = np.clip(1.0 - f_best / f_scale, 0.0, 1.0)
    gamma = 0.5 + 0.5 * progress
    
    # Multiplicative cost penalty
    # score = EI / (cost ** gamma)
    cost_penalty = np.power(cost_safe, gamma)
    score = EI / cost_penalty
    
    # Exploration bonus:
    # In regions with high sigma, there is potential for large improvement 
    # not captured by EI if mu is not yet low.
    # Add a small term proportional to sigma, also penalized by cost.
    # Use a normalized sigma to keep the bonus small relative to EI.
    sigma_max = np.max(sigma_safe) if np.max(sigma_safe) > 0 else 1.0
    # Bonus is active when EI is low (below 25th percentile)
    ei_threshold = np.percentile(EI, 25) if n > 0 else 0
    low_ei_mask = EI < ei_threshold
    
    # The bonus should be small enough not to override EI but large enough 
    # to steer towards high-variance cheap areas if EI is flat.
    # Scale factor 0.1 is a guess based on typical EI magnitudes.
    bonus = 0.1 * (sigma_safe / sigma_max)
    # Apply cost penalty to bonus as well
    bonus = bonus / cost_penalty
    
    score[low_ei_mask] += bonus[low_ei_mask]
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
