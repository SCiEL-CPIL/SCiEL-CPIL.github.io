import numpy as np
import time

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    
    # Normalized UCB
    # Exploitation: prefer low mu
    # Exploration: prefer high sigma
    
    # Handle edge cases where max == min
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    sigma_min = np.min(sigma)
    sigma_max = np.max(sigma)
    
    if mu_max > mu_min:
        mu_norm = (mu - mu_min) / (mu_max - mu_min)
    else:
        mu_norm = np.zeros_like(mu)
        
    if sigma_max > sigma_min:
        sigma_norm = (sigma - sigma_min) / (sigma_max - sigma_min)
    else:
        sigma_norm = np.zeros_like(sigma)
        
    # Score: Higher is better
    # Exploitation term: 1 - mu_norm (1 if best mu, 0 if worst mu)
    # Exploration term: beta * sigma_norm (beta if most uncertain, 0 if least)
    
    # Beta = 1.0 means equal weight to normalized exploitation and exploration
    beta = 1.0
    
    scores = (1.0 - mu_norm) + beta * sigma_norm
    
    # Deadline guard
    if time.time() - start_time > 1.0:
        # Fallback to simple exploitation
        scores = -mu
        
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
