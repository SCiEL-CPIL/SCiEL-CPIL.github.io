import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Args:
        mu: (2000,) array of GP means.
        sigma: (2000,) array of GP standard deviations.
        f_best: scalar, best observed value.
        cost: (2000,) array of evaluation costs.
        
    Returns:
        (2000,) array of scores (higher is better).
    """
    start_time = time.time()
    
    # Safety margin: ensure we finish well before 330s.
    # This function is called many times, but each call must be fast.
    # We assume the caller manages the total budget, but we add a guard here
    # just in case this specific call is taking too long due to large arrays 
    # (though 2000 is small).
    
    # 1. Sample from the posterior: y ~ N(mu, sigma^2)
    # We use a fixed seed based on the current state? No, random is better for TS.
    # However, to be deterministic if required by some frameworks, we could use 
    # a hash of mu/sigma, but standard TS uses random sampling.
    # The prompt implies a standard loop, so random sampling is expected.
    
    # Ensure sigma is non-negative and handle numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Sample a single function value for each candidate
    # y = mu + sigma * z, where z ~ N(0, 1)
    z = np.random.standard_normal(len(mu))
    y = mu + sigma_safe * z
    
    # 2. Calculate score
    # We want to minimize y, but also minimize cost.
    # Score should be higher for better candidates.
    # Better candidate = lower y, lower cost.
    # Score = -y / cost
    
    # Prevent division by zero or very small costs
    cost_safe = np.maximum(cost, 1e-8)
    
    scores = -y / cost_safe
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return scores
