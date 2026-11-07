import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Probability of Improvement (PI) for minimization
    # P(y < f_best) = Phi((f_best - mu) / sigma)
    
    scores = np.zeros_like(mu)
    
    # Mask for points with significant uncertainty
    mask = sigma > 1e-8
    
    if np.any(mask):
        mu_m = mu[mask]
        sigma_m = sigma[mask]
        
        # Compute Z
        z = (f_best - mu_m) / sigma_m
        
        # Compute PI
        pi = norm.cdf(z)
        
        scores[mask] = pi
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        return scores
        
    return scores
