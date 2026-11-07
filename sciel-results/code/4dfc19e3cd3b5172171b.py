import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # Safety margin for time, though this function is very fast
    # start_time = time.time()
    
    # Calculate Standard EI
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Calculate Normalized Exploration Term
    # We want to reward high sigma
    sigma_max = np.max(sigma_safe)
    sigma_min = np.min(sigma_safe)
    if sigma_max > sigma_min:
        exploration = (sigma_safe - sigma_min) / (sigma_max - sigma_min)
    else:
        exploration = np.zeros_like(sigma_safe)
        
    # Calculate Normalized Exploitation Term (lower mu is better)
    # We want to reward low mu
    mu_max = np.max(mu)
    mu_min = np.min(mu)
    if mu_max > mu_min:
        exploitation = (mu_max - mu) / (mu_max - mu_min) # High score for low mu
    else:
        exploitation = np.zeros_like(mu)
        
    # Blend: Weight exploration more heavily in the beginning?
    # Since we don't know the step count, we can use the magnitude of EI or sigma.
    # A fixed blend is often robust.
    # Let's try a UCB-like approach on the normalized values.
    # Score = exploitation + beta * exploration
    # Let's set beta = 2.0 to favor exploration, which helps find global minima in multimodal landscapes.
    
    beta = 2.0
    score = exploitation + beta * exploration
    
    # Alternative: Pure UCB in original scale
    # score_ucb = -mu + 2.0 * sigma_safe
    
    # The normalized version prevents one term from dominating due to scale.
    # Michalewicz values are around -1.8 to 1.
    # Sigma is likely smaller in later iterations.
    
    return score
