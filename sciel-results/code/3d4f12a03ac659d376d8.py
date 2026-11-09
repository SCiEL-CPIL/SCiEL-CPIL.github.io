import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a blend of Normalized EI and PI.
    """
    t_start = time.time()
    
    # Safety for sigma
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Normalize to [0, 1] using Min-Max
    ei_norm = (ei - np.min(ei)) / (np.max(ei) - np.min(ei) + 1e-9)
    pi_norm = (pi - np.min(pi)) / (np.max(pi) - np.min(pi) + 1e-9)
    
    # Blend EI and PI
    # Weight EI higher as it provides magnitude of improvement
    blended = 0.7 * ei_norm + 0.3 * pi_norm
    
    # Cost-aware adjustment
    cost_safe = np.maximum(cost, 1e-10)
    scores = blended / cost_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Time guard
    t_elapsed = time.time() - t_start
    if t_elapsed > 280:
        return scores
        
    return scores
