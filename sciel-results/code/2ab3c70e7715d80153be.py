import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    
    # Ensure sigma is positive to avoid division by zero in z calculation
    # The problem states sigma is a standard deviation, so it should be >= 0
    # but numerical issues might yield 0.
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI) for minimization
    # We want to maximize EI, which is E[max(0, f_best - f(x))]
    
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip z to prevent extreme values which can cause numerical instability in norm.pdf/cdf
    # z > 8 is effectively 1 for CDF and 0 for PDF
    # z < -8 is effectively 0 for CDF and very small for PDF
    z = np.clip(z, -10, 10)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: (f_best - mu) is the potential improvement
    # Phi(z) is the probability that f(x) < f_best
    # phi(z) is the standard normal PDF
    
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    scores = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure scores are finite
    # If f_best is very large or mu is very small, scores might be large but finite.
    # If there are NaNs, replace with 0.
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Deadline guard
    if time.time() - start_time > 1.0:
        # Fallback: Greedy (minimize mu, so maximize -mu)
        # This is a safe fallback that always points to the most likely best point
        scores = -mu
        
    return scores
