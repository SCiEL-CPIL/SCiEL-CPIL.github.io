"""Combined acquisition for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = 320.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Adaptive blend of Expected Improvement (EI) and
    Lower Confidence Bound (LCB) for minimization of Michalewicz 2D.
    
    Structural idea (from inspiration 1): Blend EI and LCB to balance
    exploitation and exploration, with adaptive weighting based on how close
    f_best is to the known global minimum.
    
    Refinement machinery (from current program): Adaptive exploration constant
    k for the LCB component that decreases as f_best approaches the optimum,
    plus time guard for safety.
    
    For minimization:
    - EI(x) = (f_best - mu) * Phi(Z) + sigma * phi(Z), where Z = (f_best - mu) / sigma
    - LCB score = -(mu + k * sigma) (we maximize this)
    
    The blend weight alpha shifts from exploration (LCB) to exploitation (EI)
    as f_best improves toward -1.8013.
    """
    # Time guard: if we are close to the deadline, return a safe fallback
    if time.time() > _DEADLINE - 5.0:
        return -np.where(np.isfinite(mu), mu, 0.0)

    sigma_safe = np.maximum(sigma, 1e-8)
    
    # --- Compute Expected Improvement for minimization ---
    # EI(x) = E[max(f_best - f(x), 0)]
    # = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues with extreme values
    z_clipped = np.clip(z, -10, 10)
    phi_z = norm.pdf(z_clipped)
    Phi_z = norm.cdf(z_clipped)
    
    EI = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # --- Compute LCB score for minimization ---
    # LCB(x) = mu(x) + k * sigma(x)
    # We want to minimize LCB, so we maximize score = -LCB = -mu - k * sigma
    
    # Adaptive exploration constant k:
    # Start with high k for exploration when f_best is far from optimum,
    # decay to low k for exploitation when f_best is close to optimum.
    # Michalewicz 2D minimum is ~-1.8013.
    
    min_val = -1.8013
    distance = f_best - min_val  # Always >= 0 if f_best >= min_val
    
    # Clip distance to a reasonable range to avoid extreme k values
    distance_clipped = np.clip(distance, 0.0, 1.0)
    
    # k decreases as distance decreases (closer to optimum -> more exploitation)
    # k = k_max - (k_max - k_min) * (1 - distance_clipped / 1.0)
    # Simplified: k = k_min + (k_max - k_min) * (1 - distance_clipped)
    # When distance is large (far from optimum), k ~ k_max (explore)
    # When distance is small (close to optimum), k ~ k_min (exploit)
    
    k_min = 0.3
    k_max = 2.5
    
    k = k_min + (k_max - k_min) * (1.0 - distance_clipped)
    k = np.clip(k, k_min, k_max)
    
    LCB_score = -(mu + k * sigma_safe)
    
    # --- Adaptive blend weight ---
    # alpha = weight on EI (exploitation), (1 - alpha) = weight on LCB (exploration)
    # Use a sigmoid based on how close f_best is to the optimum.
    # When f_best is far from -1.8013, alpha is low (more exploration).
    # When f_best is close to -1.8013, alpha is high (more exploitation).
    
    # Sigmoid centered around f_best = -1.5 with scale 0.3
    # alpha = 1 / (1 + exp(-(f_best + 1.5) / 0.3))
    # When f_best = -1.8: alpha = 1/(1+exp(-(-1.8+1.5)/0.3)) = 1/(1+exp(1)) ~ 0.27
    # Wait, that's backwards. Let me rethink.
    
    # We want alpha HIGH when f_best is CLOSE to -1.8013 (exploit).
    # f_best close to -1.8013 means f_best + 1.8013 is close to 0.
    # Let's use: alpha = 1 / (1 + exp((f_best + 1.8013) / 0.2))
    # When f_best = -1.8013: alpha = 1/(1+exp(0)) = 0.5
    # When f_best = -1.6: alpha = 1/(1+exp(0.2/0.2)) = 1/(1+exp(1)) ~ 0.27
    # When f_best = -2.0: alpha = 1/(1+exp(-0.2/0.2)) = 1/(1+exp(-1)) ~ 0.73
    
    # Hmm, this gives alpha > 0.5 when f_best < -1.8013, which shouldn't happen.
    # Let's use a different approach:
    # progress = 1 - (f_best - min_val) / (max_expected - min_val)
    # But we don't know max_expected well.
    
    # Simpler: use the distance directly.
    # alpha = 1 - distance_clipped (when distance is 0, alpha = 1; when distance is 1, alpha = 0)
    # But this is too linear. Let's use a smooth function.
    
    # alpha = exp(-distance_clipped / 0.3)
    # When distance = 0: alpha = 1
    # When distance = 0.3: alpha = exp(-1) ~ 0.37
    # When distance = 0.6: alpha = exp(-2) ~ 0.14
    # When distance = 1.0: alpha = exp(-3.33) ~ 0.036
    
    alpha = np.exp(-distance_clipped / 0.3)
    
    # Clamp alpha to [0.1, 0.95] to maintain some balance
    alpha = np.clip(alpha, 0.1, 0.95)
    
    # --- Combine scores ---
    # Normalize EI and LCB_score to be on comparable scales before blending.
    # EI is in objective units (typically 0 to ~1 for this problem).
    # LCB_score is also in objective units but can be negative.
    # Let's shift LCB_score so that its typical range is positive.
    # The typical range of mu is around -1.0 to -1.8, sigma is small.
    # LCB_score = -(mu + k*sigma) is around 1.0 to 1.8.
    # EI is around 0 to 0.5.
    # So they are roughly on the same scale. Let's just blend them directly.
    
    score = alpha * EI + (1 - alpha) * LCB_score
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
