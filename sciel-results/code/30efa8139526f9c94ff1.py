# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Avoid division by zero
    sigma_safe = sigma + 1e-9
    cost_safe = cost + 1e-9
    
    # Probability of Improvement
    z = (f_best - mu) / sigma_safe
    poi = norm.cdf(z)
    
    # Expected Improvement (standard)
    ei = (f_best - mu) * poi + sigma_safe * norm.pdf(z)
    
    # New Score: PoI weighted by potential improvement magnitude
    # This removes the pure exploration term (sigma * pdf) from EI
    # which might be driving search to cheap high-uncertainty corners.
    # It keeps the exploitation signal (f_best - mu) gated by PoI.
    improvement_potential = np.maximum(f_best - mu, 0)
    score = (improvement_potential * poi) / cost_safe
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
