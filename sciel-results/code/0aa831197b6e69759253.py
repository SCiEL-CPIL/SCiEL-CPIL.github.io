import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function: PI-sigma blend.
    Scores candidates based on Probability of Improvement weighted by Uncertainty.
    """
    # Deadline guard (though this function is very fast)
    start_time = time.time()
    deadline = start_time + 2.0  # 2 seconds is plenty for 2000 points
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score: (f_best - mu) / sigma
    # Higher z means mu is lower (better) than f_best
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement (PI)
    # P(f(x) < f_best)
    pi = norm.cdf(z)
    
    # PI-sigma blend
    # Weights the probability of improvement by the standard deviation (uncertainty).
    # This balances exploitation (high PI) with exploration (high sigma).
    score = pi * sigma_safe
    
    # Handle any numerical issues
    score = np.nan_to_num(score, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # Check time (unlikely to trigger)
    if time.time() > deadline:
         # Return the best available result immediately
         pass 
         
    return score
