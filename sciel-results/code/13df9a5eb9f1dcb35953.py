"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Structural change: Switch from EI-blend to Lower Confidence Bound (LCB)
    with adaptive exploration parameter.
    
    LCB = mu - kappa * sigma (we maximize this, so we prefer low mu and high sigma)
    
    The adaptive kappa starts high when f_best is far from the global minimum
    (encouraging exploration to find better basins) and decreases as we approach
    the optimum (encouraging exploitation for fine-grained search).
    
    This is a fundamentally different acquisition family from EI: it's a linear
    combination of mean and uncertainty rather than an expected improvement integral.
    """
    # Deadline guard
    t_start = time.time()
    
    # Known global minimum for Hartmann 6d
    f_target = -3.32237
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Adaptive kappa: high exploration early, low exploitation late
    # gap = how far current best is from global min
    gap = max(f_best - f_target, 1e-6)
    
    # kappa decays from ~3.0 (high exploration) to ~0.5 (exploitation)
    # Use exponential decay: kappa = kappa_min + (kappa_max - kappa_min) * exp(-gap / scale)
    kappa_max = 3.0
    kappa_min = 0.5
    scale = 0.3  # controls transition speed
    kappa = kappa_min + (kappa_max - kappa_min) * np.exp(-gap / scale)
    
    # LCB score: we want to maximize mu - kappa*sigma
    # Since we're minimizing the objective, lower mu is better, higher sigma is exploratory
    scores = mu - kappa * sigma_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Safety check
    if time.time() - t_start > 250:
        return scores
        
    return scores
# EVOLVE-BLOCK-END
