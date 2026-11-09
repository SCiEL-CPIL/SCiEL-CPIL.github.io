"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Ensure sigma is positive to avoid division errors
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate Expected Improvement (EI)
    # Higher EI means better expected reduction in f_best
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Calculate Exploration Score (Standard Deviation)
    # Higher sigma means higher uncertainty
    exp_score = sigma
    
    # Normalize EI and Exploration scores to have zero mean and unit variance
    # This allows for a fair linear combination regardless of scale
    ei_mean = np.mean(ei)
    ei_std = np.std(ei)
    if ei_std < 1e-9:
        ei_norm = np.zeros_like(ei)
    else:
        ei_norm = (ei - ei_mean) / ei_std
        
    exp_mean = np.mean(exp_score)
    exp_std = np.std(exp_score)
    if exp_std < 1e-9:
        exp_norm = np.zeros_like(exp_score)
    else:
        exp_norm = (exp_score - exp_mean) / exp_std
        
    # Determine the weight for Exploitation (EI) vs Exploration (Sigma)
    # w is the weight for EI.
    # When f_best is large (early), w is small -> favor Exploration.
    # When f_best is small (late), w is large -> favor Exploitation.
    # Using an exponential decay ensures a smooth transition.
    alpha = 0.1
    w = np.exp(-alpha * f_best)
    
    # Combine the normalized scores
    # scores = w * Exploitation + (1 - w) * Exploration
    scores = w * ei_norm + (1 - w) * exp_norm
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
