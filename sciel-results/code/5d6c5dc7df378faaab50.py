import numpy as np
import time

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    
    # Compute scores using Thompson Sampling
    z = np.random.randn(mu.shape)
    y = mu + sigma * z
    scores = -y
    
    # Safety margin: if we are taking too long (unlikely), return zeros
    # to avoid timeout.
    if time.time() - start_time > 290:
        return np.zeros_like(mu)
        
    # Ensure finite scores
    scores[~np.isfinite(scores)] = -1e10
    
    return scores
