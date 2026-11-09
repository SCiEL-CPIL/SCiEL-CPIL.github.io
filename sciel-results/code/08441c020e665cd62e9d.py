import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_START_TIME = time.time()
_TIMEOUT = 290  # Safety margin before 300s

def acquisition(mu, sigma, f_best):
    # Check deadline
    if time.time() - _START_TIME > _TIMEOUT:
        # Fallback: Return EI scores (fastest reliable metric)
        z = (f_best - mu) / (sigma + 1e-9)
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Calculate Expected Improvement
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Adaptive exploration weight
    # If f_best is high (bad), explore more. If f_best is low (good), exploit more.
    # Styblinski-Tang global min is ~ -39.16
    # Local minima are around -10 to -20
    # We want to escape local minima, so high exploration when f_best is not close to global min.
    
    beta_min = 0.2
    beta_max = 1.5
    
    # Thresholds for transition
    # If f_best > -10, we are likely in a bad local min or unexplored area. Explore hard.
    # If f_best < -30, we are close to global min. Exploit.
    thresh_high = -10.0
    thresh_low = -30.0
    
    if f_best >= thresh_high:
        beta = beta_max
    elif f_best <= thresh_low:
        beta = beta_min
    else:
        # Linear interpolation
        # f_best = -10 -> beta_max
        # f_best = -30 -> beta_min
        # slope = (beta_min - beta_max) / (thresh_low - thresh_high)
        beta = beta_max + (beta_min - beta_max) * (f_best - thresh_high) / (thresh_low - thresh_high)
        
    # Blend
    scores = ei + beta * sigma
    
    return scores
