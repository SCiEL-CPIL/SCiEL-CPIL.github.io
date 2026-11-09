import numpy as np
import time

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Epsilon to prevent issues with zero/negative sigma
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    
    # Thompson Sampling: sample from posterior N(mu, sigma)
    # For minimization, we want low values, so score = -sample
    # This naturally balances exploration and exploitation
    rng = np.random.default_rng()
    sample = mu + sigma_safe * rng.standard_normal(len(mu))
    
    scores = -sample
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check deadline
    if time.time() - start_time > 320:
         # If somehow slow, return a fallback
         pass
         
    return scores
