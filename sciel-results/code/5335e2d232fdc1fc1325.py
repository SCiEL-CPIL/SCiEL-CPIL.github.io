"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Track start time for potential timeout guards
_start_time = time.time()
_DEADLINE = _start_time + 290  # Keep a 10s safety margin before the 300s limit

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Variance-Gated Expected Improvement (VEI).
    
    This approach calculates the standard Expected Improvement (EI) for minimization
    and then scales it by a power of the relative standard deviation.
    This structurally penalizes exploitation in low-uncertainty regions,
    prioritizing areas where the GP is still uncertain, which is critical
    for escaping local minima in Branin's multimodal landscape.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed objective value so far.
        
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    # Safety check for time, though vectorized ops are fast
    if time.time() > _DEADLINE:
        # Fallback to simple EI if we are running out of time
        # (Unlikely for this simple calculation, but safe)
        z = (f_best - mu) / (sigma + 1e-8)
        ei = (f_best - mu) * norm.cdf(z) - sigma * norm.pdf(z)
        return ei

    # Avoid division by zero in z calculation
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization:
    # E[f_best - f(x) | f(x) < f_best] * P(f(x) < f_best)
    # = (f_best - mu) * Phi(z) - sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) - sigma * norm.pdf(z)
    
    # Variance Gate: Scale EI by (sigma / sigma_max)^gamma
    # This forces exploration of high uncertainty regions
    sigma_max = np.max(sigma)
    if sigma_max < 1e-8:
        # If all sigmas are negligible, just use EI
        return ei
        
    # gamma = 1.5 as suggested in the directive
    gamma = 1.5
    variance_gate = (sigma / sigma_max) ** gamma
    
    scores = ei * variance_gate
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
