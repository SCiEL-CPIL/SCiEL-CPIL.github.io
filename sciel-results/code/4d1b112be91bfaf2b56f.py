"""Acquisition function for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = None


def acquisition(mu, sigma, f_best):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Time guard: if we're running too long, return a safe fallback
    elapsed = time.time() - _start_time
    if elapsed > 280:
        # Return exploration-based scores to avoid timeout
        return -mu + 0.1 * sigma
    
    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Normalize mu and sigma for numerical stability
    mu_min = np.min(mu)
    mu_range = np.max(mu) - mu_min + 1e-10
    
    # UCB component: balance exploitation and exploration
    # Use a dynamic beta that decreases over time
    beta = 0.5
    
    # UCB score (higher is better for exploration)
    ucb_score = -mu + beta * sigma_safe
    
    # Modified EI with exploration bonus
    # Standard EI
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a small exploration bonus based on uncertainty
    # Points with higher uncertainty get a small boost
    exploration_bonus = 0.1 * sigma_safe / (sigma_safe.max() + 1e-10)
    
    # Combine: use EI as primary, UCB as secondary
    # Normalize both to [0, 1] range for combination
    ei_min, ei_max = np.min(ei), np.max(ei)
    ei_range = ei_max - ei_min + 1e-10
    ei_norm = (ei - ei_min) / ei_range
    
    ucb_min, ucb_max = np.min(ucb_score), np.max(ucb_score)
    ucb_range = ucb_max - ucb_min + 1e-10
    ucb_norm = (ucb_score - ucb_min) / ucb_range
    
    # Weighted combination: 70% EI, 30% UCB
    score = 0.7 * ei_norm + 0.3 * ucb_norm
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
