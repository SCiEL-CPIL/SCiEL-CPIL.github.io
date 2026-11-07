import numpy as np
import time
from scipy.stats import norm

_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best):
    global _start_time, _call_count
    if _start_time is None:
        _start_time = time.time()
    
    _call_count += 1
    
    # Deadline guard
    if time.time() - _start_time > 325:
        # Fallback to EI if time is tight
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.where(np.isfinite(ei), ei, 0.0)
        return ei
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Structural Change: Use Expected Improvement (EI) which is robust for multimodal landscapes.
    # EI directly optimizes the expected gain over the current best, balancing exploration and exploitation
    # through the probabilistic structure of the GP posterior.
    
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # For minimization, we maximize EI.
    
    z = (f_best - mu) / sigma_safe
    
    # Calculate standard normal CDF and PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure finite scores
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
