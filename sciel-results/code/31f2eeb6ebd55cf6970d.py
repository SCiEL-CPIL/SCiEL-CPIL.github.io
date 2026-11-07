# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import erf
import time

# Global timer for deadline guard
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Args:
        mu: (N,) array of GP mean predictions.
        sigma: (N,) array of GP standard deviations.
        f_best: float, best observed value so far.
        cost: (N,) array of costs for each candidate.
    
    Returns:
        (N,) array of acquisition scores (higher is better).
    """
    global _start_time
    
    # Initialize timer on first call
    if _start_time is None:
        _start_time = time.time()
        
    elapsed = time.time() - _start_time
    
    # Deadline guard: if we are close to the 300s limit, switch to a fast, safe mode
    # to ensure we return valid scores and allow the loop to finish or terminate gracefully.
    # 290s is a safety margin.
    if elapsed > 290:
        sigma_safe = np.maximum(sigma, 1e-10)
        cost_safe = np.maximum(cost, 1e-10)
        # Maximize exploration relative to cost as a safe fallback
        scores = sigma_safe / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)

    N = len(mu)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Standardized distance
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues in erf/exp
    z = np.clip(z, -10.0, 10.0)
    
    # Calculate CDF and PDF using erf for speed and stability
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    # phi(z) = exp(-0.5 * z^2) / sqrt(2 * pi)
    
    sqrt_2 = 1.4142135623730951
    sqrt_2_pi = 2.5066282746310002
    
    cdf_z = 0.5 * (1.0 + erf(z / sqrt_2))
    pdf_z = np.exp(-0.5 * z * z) / sqrt_2_pi
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei_term = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure EI is non-negative
    ei_term = np.maximum(ei_term, 0.0)
    
    # Cost-aware: divide by cost
    cost_safe = np.maximum(cost, 1e-10)
    
    scores = ei_term / cost_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
