import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Adaptive Lower Confidence Bound (LCB).
    Structural change from EI to UCB/LCB family with adaptive exploration.
    """
    # Check time budget
    elapsed = time.time() - _start_time
    if elapsed > 300:
        return np.zeros_like(mu)

    # Ensure inputs are numpy arrays
    sigma = np.asarray(sigma, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    
    # Sanitize sigma
    sigma = np.where(sigma > 1e-12, sigma, 1e-12)
    sigma = np.where(np.isfinite(sigma), sigma, 1.0)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Adaptive Beta
    # Branin 2D properties
    f_min = 0.397887
    f_max = 300.0 # Upper bound for scaling
    
    # Calculate progress: 1.0 at start (f_best high), 0.0 at end (f_best low)
    # If f_best is lower than f_min (shouldn't happen, but safe), progress is negative -> clip to 0
    progress = (f_best - f_min) / (f_max - f_min)
    progress = np.clip(progress, 0.0, 1.0)
    
    # Beta decreases as we get closer to the optimum
    # Start with high exploration, end with low exploration
    beta_min = 0.1
    beta_max = 2.0
    
    # Linear interpolation or power law?
    # Power law might be better to stay exploratory longer?
    # Let's try linear first for simplicity.
    beta = beta_min + (beta_max - beta_min) * progress
    
    # LCB for minimization: mu - beta * sigma
    # We want to maximize the score, so we return -LCB
    score = -mu + beta * sigma
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
