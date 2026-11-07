import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best):
    # Deadline guard: if we're close to the time limit, return a safe fallback
    if time.time() - _start_time > 310:
        return -mu

    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-8)

    # Adaptive Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    #
    # We use an adaptive exploration parameter xi that scales with the
    # current uncertainty to balance exploration and exploitation.
    #
    # Strategy: xi adapts based on the spread of sigma values.
    # When uncertainty is high (early in optimization), use larger xi
    # to encourage exploration. When uncertainty is low, use smaller xi
    # to focus on exploitation.
    
    # Compute a measure of overall uncertainty
    sigma_mean = np.mean(sigma_safe)
    sigma_std = np.std(sigma_safe)
    
    # Adaptive xi: scale with the mean sigma but cap it
    # Use a fraction of the typical uncertainty
    # This helps escape local optima when uncertainty is high
    xi = 0.05 * sigma_mean + 0.01
    
    # Clip xi to reasonable bounds
    xi = np.clip(xi, 0.0, 0.5)
    
    # Compute z-score
    z = (f_best - xi - mu) / sigma_safe
    
    # Compute EI scores
    # For numerical stability, handle large positive and negative z values
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei_scores = (f_best - xi - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure finite scores
    ei_scores = np.where(np.isfinite(ei_scores), ei_scores, -1e10)
    
    return ei_scores
