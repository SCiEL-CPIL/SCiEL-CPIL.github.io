"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 320.0  # safety margin before 330s kill


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a structural hybrid:
    - Early stage (high f_best): LCB with high kappa for exploration
    - Late stage (low f_best): EI for exploitation
    
    The transition is smooth and based on f_best relative to the known
    range of the Branin function.
    """
    # Time guard
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Fallback: return EI scores
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        return (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute EI (Expected Improvement for minimization)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Compute LCB (Lower Confidence Bound for minimization)
    # For minimization, we want low values, so LCB score = -mu + kappa * sigma
    # Higher score means more promising
    
    # Stage indicator: how far are we from the initial f_best?
    # Branin ranges from ~0.4 to ~30 in typical random starts
    # Normalize f_best to [0, 1] where 0 = very good (exploit), 1 = very bad (explore)
    f_max_est = 50.0
    stage = np.clip(f_best / f_max_est, 0.0, 1.0)
    
    # Kappa for LCB: high when exploring, low when exploiting
    # Start at 2.5 (strong exploration), end at 0.3 (strong exploitation)
    kappa = 0.3 + 2.2 * stage
    
    # LCB score (higher is better for selection)
    lcb_score = -mu + kappa * sigma_safe
    
    # Blend EI and LCB based on stage
    # When stage is high (early), weight LCB more
    # When stage is low (late), weight EI more
    
    # Use a smooth blending function
    # weight_lcb = stage^2 (quadratic to emphasize exploration early)
    # weight_ei = 1 - stage^2
    weight_lcb = stage ** 2
    weight_ei = 1.0 - weight_lcb
    
    # Normalize both scores to [0, 1] for fair blending
    ei_min, ei_max = np.min(ei), np.max(ei)
    lcb_min, lcb_max = np.min(lcb_score), np.max(lcb_score)
    
    if ei_max - ei_min > 1e-9:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.zeros_like(ei)
    
    if lcb_max - lcb_min > 1e-9:
        lcb_norm = (lcb_score - lcb_min) / (lcb_max - lcb_min)
    else:
        lcb_norm = np.zeros_like(lcb_score)
    
    # Blend
    score = weight_lcb * lcb_norm + weight_ei * ei_norm
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
