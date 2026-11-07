import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (for minimization)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # UCB for minimization: mu - kappa * sigma
    # Adaptive kappa based on how concentrated mu is
    mu_std = np.std(mu)
    mu_range = np.max(mu) - np.min(mu)
    
    # If mu is spread out, we're still exploring -> use larger kappa
    # If mu is concentrated, we're exploiting -> use smaller kappa
    # Normalize mu_range to [0, 1] roughly
    kappa = 0.5 + 1.5 * np.clip(mu_range / (2 * mu_std + 1e-9), 0, 1)
    
    ucb = mu - kappa * sigma
    
    # Cost-normalized scores
    ei_score = ei / cost
    ucb_score = ucb / cost
    
    # Blend: when GP is uncertain (high sigma relative to mu spread), use more UCB
    sigma_mean = np.mean(sigma)
    blend = np.clip(sigma_mean / (mu_std + 1e-9), 0, 1)
    
    score = (1 - blend) * ei_score + blend * ucb_score
    
    return score
