import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_TIME_LIMIT = 280.0

def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        return -mu
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # Handle edge case where f_best is very small
    # If f_best is 0, we are done, but let's be safe.
    if f_best <= 1e-9:
        # Pure exploitation
        return -mu
    
    # Probability of Improvement (POI)
    # P(f < f_best) = Phi((f_best - mu) / sigma)
    # Where Phi is the CDF of the standard normal distribution
    # This ranks candidates by their likelihood of beating the incumbent
    
    # Avoid division by zero
    safe_sigma = np.maximum(sigma, 1e-10)
    
    # Compute z-scores: how many standard deviations f_best is above mu
    z = (f_best - mu) / safe_sigma
    
    # Compute the CDF of the standard normal at z
    # This gives P(f < f_best) for each candidate
    scores = norm.cdf(z)
    
    # Ensure finite scores
    # If any NaNs or Infs appear, replace with a very low score
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    return scores
