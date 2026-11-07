# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Variance-Gated Expected Improvement (VEI) acquisition function.
    
    Scales the standard Expected Improvement by (sigma / sigma_max)^gamma
    to penalize exploitation in low-uncertainty regions and encourage
    exploration in areas where the GP is still uncertain.
    
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
    
    # Ensure sigma is strictly positive for numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Variance gate: scale EI by (sigma / sigma_max)^gamma
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-12:
        gamma = 1.5
        variance_gate = (sigma_safe / sigma_max) ** gamma
    else:
        variance_gate = np.ones_like(sigma_safe)
    
    scores = ei * variance_gate
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
