"""Acquisition function for Bayesian optimization of hartmann 3d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def _sanitize(scores):
    """Ensure scores are finite and of the correct shape."""
    if scores.shape != (2000,):
        scores = np.resize(scores, (2000,))
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    deadline = start_time + 280.0  # 300s limit, leave 20s safety margin
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-8)
    mu_safe = np.nan_to_num(mu, nan=f_best, posinf=f_best, neginf=f_best)
    f_best_safe = np.nan_to_num(f_best, nan=-1e6, posinf=1e6, neginf=-1e6)
    
    # Compute Expected Improvement (EI) for minimization
    z = (f_best_safe - mu_safe) / sigma_safe
    z = np.clip(z, -10, 10)
    ei_scores = (f_best_safe - mu_safe) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Compute Lower Confidence Bound (LCB) for minimization
    # LCB = mu - beta * sigma, we want to maximize -LCB = -mu + beta * sigma
    # Use adaptive beta based on the current uncertainty level
    avg_sigma = np.mean(sigma_safe)
    # Start with moderate exploration, increase if uncertainty is high
    beta = 1.0 + 2.0 * np.exp(-avg_sigma * 5.0)
    lcb_scores = -mu_safe + beta * sigma_safe
    
    # Normalize both scores to [0, 1] range for combination
    ei_min, ei_max = np.min(ei_scores), np.max(ei_scores)
    if ei_max > ei_min + 1e-12:
        ei_norm = (ei_scores - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.zeros_like(ei_scores)
    
    lcb_min, lcb_max = np.min(lcb_scores), np.max(lcb_scores)
    if lcb_max > lcb_min + 1e-12:
        lcb_norm = (lcb_scores - lcb_min) / (lcb_max - lcb_min)
    else:
        lcb_norm = np.zeros_like(lcb_scores)
    
    # Adaptive weighting: emphasize EI when we have good estimates (low uncertainty),
    # emphasize LCB when uncertainty is high (early in optimization)
    # Use a smooth transition based on average sigma
    weight_ei = 1.0 / (1.0 + np.exp(-3.0 * (avg_sigma - 0.05)))
    weight_lcb = 1.0 - weight_ei
    
    # Combine: higher score means better candidate
    final_scores = weight_ei * ei_norm + weight_lcb * lcb_norm
    
    # Check time
    if time.time() - start_time > deadline - start_time:
        # Fallback to pure EI if hybrid takes too long
        z = (f_best_safe - mu_safe) / sigma_safe
        z = np.clip(z, -10, 10)
        ei_scores = (f_best_safe - mu_safe) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        final_scores = ei_scores

    # Ensure finite
    final_scores = _sanitize(final_scores)
    
    return final_scores
# EVOLVE-BLOCK-END
