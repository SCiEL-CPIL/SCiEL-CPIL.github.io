import numpy as np
import time

_start_time = time.time()
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best):
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Fallback: return pure PMM scores
        scores = -np.asarray(mu, dtype=np.float64)
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1.0)
    
    # Normalize sigma to [0, 1] for adaptive beta
    sigma_max = np.max(sigma)
    if sigma_max > 0:
        sigma_norm = sigma / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma)
    
    # Adaptive UCB: balance exploration and exploitation
    # Early on (high uncertainty), explore more; later, exploit more
    # Use a beta that decreases as we get more confident
    # The idea: score = -mu + beta * sigma
    # We want beta to be large when uncertainty is high relative to the scale of mu
    
    # Scale beta based on the ratio of uncertainty to the current best
    # If f_best is close to 0, we're doing well and should be more exploitative
    # If f_best is far from 0, we should explore more
    
    # Normalize f_best to get a sense of how far we are from optimal
    # sphere 1d on [-5, 5] has max value 25, so f_best in [0, 25]
    # Use a sigmoid-like function to get beta in a reasonable range
    
    # Start with a base beta that balances exploration and exploitation
    # For 1D sphere, a good balance is around beta = 0.5-1.0
    # We'll adapt beta based on how much uncertainty remains
    
    # When sigma is large relative to the scale of the problem, explore more
    # When sigma is small, exploit the mean
    
    # Use a geometric mean approach: score = -mu + beta * sigma
    # where beta is chosen to balance the two terms
    
    # Normalize mu and sigma to have comparable scales
    mu_scale = np.std(mu) + 1e-10
    sigma_scale = np.std(sigma) + 1e-10
    
    # Adaptive beta: higher when uncertainty is more variable
    # This encourages exploration in regions where the GP is uncertain
    beta = 0.7 + 0.3 * (sigma_scale / (mu_scale + sigma_scale + 1e-10))
    
    # Compute UCB scores
    scores = -mu + beta * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
