import numpy as np
from scipy.stats import norm
import time

# Global time tracking for deadline guard
_start_time = time.time()
_TIME_LIMIT = 300.0  # Safety margin before 330s

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI) 
    and Probability of Improvement (PI).
    """
    # Check deadline
    if time.time() - _start_time > _TIME_LIMIT:
        # Return a safe, fast score to finish quickly
        # Higher score = more likely to be picked. 
        # If we are out of time, we just want to return something valid.
        # Using 1/cost ensures finite and positive scores.
        return 1.0 / (cost + 1e-9)

    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Probability of Improvement (PI)
    pi = norm.cdf(z)
    
    # Hybrid Score: EI + alpha * PI * sigma
    # Scaling PI by sigma brings it to the same units as EI (objective space)
    alpha = 0.1
    score = ei + alpha * pi * sigma_safe
    
    # Cost-aware normalization
    # Divide by cost. Cost is 1 + mean(x_scaled). 
    # This favors points that are both promising and cheap.
    safe_cost = np.maximum(cost, 1e-9)
    final_scores = score / safe_cost
    
    # Ensure finite scores
    final_scores = np.nan_to_num(final_scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return final_scores
