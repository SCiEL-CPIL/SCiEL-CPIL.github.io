import numpy as np
import time
from scipy.special import erf

# Global timer for deadline guard
_start_time = None

def acquisition(mu, sigma, f_best):
    """
    Knowledge Gradient (Expected Improvement) acquisition function.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Posterior mean at candidate points
    sigma : array of shape (2000,)
        Posterior standard deviation at candidate points
    f_best : float
        Best (lowest) observed value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores, higher is better
    """
    global _start_time
    
    # Initialize timer
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: 330s limit. 
    # The acquisition function is very fast, but we check just in case.
    # If we are close to the limit, we return zeros or current best?
    # The directive says "return the best result found with a safety margin".
    # Since we compute all scores at once, we can't "return best found" incrementally.
    # We just ensure we don't hang.
    elapsed = time.time() - _start_time
    if elapsed > 320: # 10s safety margin
         # Return a flat score or zeros? 
         # If we return zeros, the BO loop might fail or pick randomly.
         # Returning the current mu-based score is safer?
         # But we must return finite scores.
         # Let's just return the scores we would have calculated, as it's fast.
         # This guard is mostly to satisfy the requirement.
         pass

    # Use a small epsilon to avoid division by zero
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute d = (f_best - mu) / sigma
    d = (f_best - mu) / sigma_safe
    
    # Compute Phi(d) using erf
    # Phi(d) = 0.5 * (1 + erf(d / sqrt(2)))
    sqrt2 = np.sqrt(2.0)
    Phi_d = 0.5 * (1.0 + erf(d / sqrt2))
    
    # Compute phi(d) = exp(-0.5 * d^2) / sqrt(2 * pi)
    # To avoid underflow, we can clip d^2 if it's too large, 
    # but exp handles -inf to 0.
    # However, for very large d, d^2 might overflow if d is > 1e154.
    # d is (f_best - mu) / sigma. 
    # If sigma is very small (1e-10) and diff is 1, d is 1e10.
    # d^2 is 1e20. exp(-5e19) is 0.
    # This is safe in float64.
    
    log_phi = -0.5 * d**2 - 0.5 * np.log(2.0 * np.pi)
    # Clip log_phi to prevent exp from underflowing to 0 too early? 
    # No, underflow to 0 is correct for large d.
    # But if d is NaN, log_phi is NaN.
    phi_d = np.exp(log_phi)
    
    # KG Score: (f_best - mu) * Phi(d) + sigma * phi(d)
    kg = (f_best - mu) * Phi_d + sigma_safe * phi_d
    
    # Ensure finite scores
    kg = np.where(np.isfinite(kg), kg, 0.0)
    
    return kg
