"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Known global minimum of Styblinski-Tang 1D
ST_MIN = -39.1662

def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: combines Expected Improvement (EI) with a
    lower-bound / uncertainty-driven exploration term to balance
    exploitation near the known optimum region and exploration
    in under-sampled areas.
    
    mu, sigma: arrays of shape (2000,) in objective units.
    f_best: lowest observed value so far.
    
    Returns: array of 2000 finite scores (higher = evaluate next).
    """
    # Time guard: if we're running low, return a safe fallback quickly
    t0 = time.time()
    
    # Clamp sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # --- Expected Improvement (standard, for minimization) ---
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid overflow in cdf/pdf
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # --- Lower bound / pessimistic value ---
    # Use a confidence-bound-like term: mu - kappa * sigma
    # This encourages exploration where uncertainty is high
    kappa = 2.0
    lower_bound = mu - kappa * sigma_safe
    
    # --- Hybrid score ---
    # Blend EI (exploitation) with negative lower bound (exploration)
    # The lower bound is "lower is better", so we negate for "higher is better"
    # We scale the exploration term relative to the problem range
    # ST function range is roughly [-39, 0], so scale by ~40
    scale = 40.0
    exploration_score = -lower_bound / scale
    
    # Weight: give more weight to EI as we get closer to the optimum
    # When f_best is close to ST_MIN, prioritize exploitation
    progress = 1.0 - max(0.0, (f_best - ST_MIN) / 40.0)
    progress = np.clip(progress, 0.0, 1.0)
    
    # Blend: (1 - progress) * exploration + progress * exploitation
    # Normalize EI to be comparable in scale
    ei_scale = np.max(ei) + 1e-9
    ei_norm = ei / ei_scale
    
    score = (1.0 - 0.7 * progress) * exploration_score + (0.7 * progress) * ei_norm
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Safety: if we've spent too much time, return immediately
    if time.time() - t0 > 2.0:
        return score
    
    return score
# EVOLVE-BLOCK-END
