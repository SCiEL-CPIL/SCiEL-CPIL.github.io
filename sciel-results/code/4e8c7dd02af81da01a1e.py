# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function: Expected Improvement per unit cost.
    Optimized for numerical stability and speed.
    """
    # Deadline guard
    if not hasattr(acquisition, '_start_time'):
        acquisition._start_time = time.time()
    
    elapsed = time.time() - acquisition._start_time
    if elapsed > 325.0:  # Safety margin
        # Fallback: simple heuristic
        # Prefer low mu and low cost
        return -mu / (cost + 1e-9)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Calculate z-score
    # To avoid overflow in (f_best - mu) / sigma_safe if sigma is tiny and diff is large
    # But sigma is clipped to 1e-12.
    z = (f_best - mu) / sigma_safe
    
    # Calculate EI
    # Standard formula: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # For numerical stability, we can use the fact that for z > 0, 
    # EI is well-behaved. For z < 0, EI is small.
    
    # Clip z to a reasonable range to avoid numerical issues in cdf/pdf
    # norm.cdf(50) is 1.0, norm.pdf(50) is ~0.
    z_clipped = np.clip(z, -50.0, 50.0)
    
    # Use scipy's norm functions
    cdf_z = norm.cdf(z_clipped)
    pdf_z = norm.pdf(z_clipped)
    
    # Compute EI
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # EI should be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score
    cost_safe = np.maximum(cost, 1e-9)
    score = ei / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
