import numpy as np
import time

_START_TIME = time.time()

def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() - _START_TIME > 320:
        return np.zeros_like(mu)
    
    # Pareto-frontier guided exploration
    # Step 1: Identify the anchor candidate with the lowest predicted mean (mu)
    anchor_idx = np.argmin(mu)
    anchor_mu = mu[anchor_idx]
    anchor_sigma = sigma[anchor_idx]
    
    # Step 2: Compute normalized distance from anchor in prediction space
    # Distance combines both mean and uncertainty dimensions
    mu_range = np.max(mu) - np.min(mu)
    if mu_range < 1e-12:
        mu_range = 1e-12
    
    sigma_range = np.max(sigma) - np.min(sigma)
    if sigma_range < 1e-12:
        sigma_range = 1e-12
    
    # Normalized Euclidean distance in (mu, sigma) space from anchor
    mu_dist = np.abs(mu - anchor_mu) / mu_range
    sigma_dist = np.abs(sigma - anchor_sigma) / sigma_range
    dist = np.sqrt(mu_dist ** 2 + sigma_dist ** 2)
    
    # Step 3: Score as product of standard deviation and normalized distance
    # This prioritizes high-uncertainty regions that are far from the exploitation anchor
    scores = sigma * dist
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
