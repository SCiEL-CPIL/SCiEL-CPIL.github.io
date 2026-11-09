import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.nan_to_num(sigma, nan=1e-9, posinf=1e6, neginf=1e-9)
    
    # Ensure mu is finite
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Ensure cost is positive and finite
    cost = np.maximum(cost, 1e-9)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1.0)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues in norm.cdf and norm.pdf
    z = np.clip(z, -50, 50)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add an uncertainty bonus to encourage exploration
    # This balances exploitation (EI) and exploration (sigma)
    # The coefficient 0.1 is a small nudge to favor high variance regions slightly more
    uncertainty_bonus = 0.1 * sigma
    
    # Cost-aware score: higher is better
    scores = (ei + uncertainty_bonus) / cost
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Check time
    if time.time() - start_time > 290.0:
        return scores
    
    return scores
