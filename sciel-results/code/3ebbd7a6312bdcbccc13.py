import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 280.0

def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        return -mu
    
    # Ensure sigma is safe
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Quantile-normalize sigma using median
    med_sigma = np.median(sigma_safe)
    if med_sigma < 1e-6:
        med_sigma = 1e-6
        
    sigma_norm = sigma_safe / med_sigma
    
    # Structural Change: Logarithmic exploration term
    # This dampens the influence of extremely high uncertainty values
    # compared to the linear term in the previous version.
    explore_term = np.log1p(sigma_norm)
    
    # Dynamic exploration weight scaled by f_best
    lambda_ts = f_best
    
    # Score: Exploitation (-mu) + Exploration (lambda * log(1 + sigma_norm))
    score = -mu + lambda_ts * explore_term
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
