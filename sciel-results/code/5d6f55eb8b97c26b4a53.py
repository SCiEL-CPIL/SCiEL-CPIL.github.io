import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Deadline guard
    start = time.time()
    limit = 1.0 # 1 second limit per call
    
    # Ensure sigma is non-negative for safety
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI) for minimization
    # Improvement is f_best - f(x)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    scores = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Fallback if something goes wrong (e.g. NaNs)
    if not np.all(np.isfinite(scores)):
        scores = np.where(np.isfinite(scores), scores, 0.0)
        
    return scores
