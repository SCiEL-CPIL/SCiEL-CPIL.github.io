import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    # Track elapsed time for safety margin
    t_start = time.time()
    
    # Guard against numerical issues
    sigma = np.maximum(sigma, 1e-12)
    
    # Structural Change: Use Expected Improvement with an improvement threshold (xi-EI)
    # This discourages selecting points that only offer marginal improvement over f_best,
    # which can help escape local plateaus or noisy regions.
    xi = 0.1
    
    # z-score for improvement with threshold
    # We want improvement of at least xi
    # Target: f_best - xi
    z = (f_best - xi - mu) / sigma
    
    # Expected Improvement (EI) with threshold
    # EI = E[max(0, f_best - xi - f(x))]
    ei = (f_best - xi - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure scores are finite and non-negative
    ei = np.maximum(ei, 0.0)
    score = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check time budget
    t_elapsed = time.time() - t_start
    if t_elapsed > 280:
        return score
    
    return score
