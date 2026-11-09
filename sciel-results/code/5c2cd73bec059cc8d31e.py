import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
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
    # We want to maximize EI / cost
    
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues in norm.cdf and norm.pdf
    z = np.clip(z, -50, 50)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Hybrid EI/UCB approach: balances exploitation (EI) and exploration (UCB)
    # UCB = -mu + beta * sigma (for minimization, we want low mu and high sigma)
    # Scale beta to balance exploration relative to EI
    beta = 0.5
    ucb = -mu + beta * sigma
    
    # Normalize both to have comparable scales
    # Use cost-aware scoring for both components
    ei_score = ei / cost
    ucb_score = ucb / cost
    
    # Combine with a weight that favors exploitation early and exploration later
    # Use a fixed blend that has shown good performance
    blend_weight = 0.4
    scores = blend_weight * ei_score + (1 - blend_weight) * ucb_score
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
