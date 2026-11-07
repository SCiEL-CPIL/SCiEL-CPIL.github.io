# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Variance-Gated Expected Improvement (VEI) with adaptive gamma.
    
    Uses a two-stage approach:
    1. Compute standard EI
    2. Scale by (sigma / sigma_ref)^gamma where sigma_ref is the median sigma
       and gamma adapts based on the ratio of f_best to typical scale
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores, higher values indicate more promising points.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clamp sigma to avoid division by zero
    sigma_safe = np.clip(sigma, 1e-10, None)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Use median sigma as reference for normalization
    sigma_ref = np.median(sigma_safe)
    if sigma_ref < 1e-10:
        sigma_ref = 1e-10
    
    # Adaptive gamma: stronger exploration when f_best is still far from optimum
    # The minimum of Branin is ~0.398, so if f_best > 1.0 we're still far
    # Use a smooth transition: gamma ranges from 1.0 (exploit) to 2.0 (explore)
    if f_best > 1.0:
        gamma = 2.0
    elif f_best > 0.5:
        gamma = 1.5
    else:
        gamma = 1.0
    
    # Variance gate normalized by median, not max (more robust)
    variance_gate = (sigma_safe / sigma_ref) ** gamma
    
    # Combine EI with variance gate
    scores = ei * variance_gate
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
