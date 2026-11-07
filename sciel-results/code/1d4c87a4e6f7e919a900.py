import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    t0 = time.time()
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI)
    z_ei = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma * norm.pdf(z_ei)
    
    # Add small exploration bonus to EI
    score = ei + 0.1 * sigma
    
    # Safety: ensure finite
    if not np.all(np.isfinite(score)):
        score = ei
    
    if time.time() - t0 > 280:
        pass
    
    return score
