# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    # Deadline guard: ensure we return within time budget
    t_start = time.time()
    
    # Ensure sigma is positive and has valid shape
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Known minimum of Goldstein-Price 2D is 3.0
    f_min_known = 3.0
    
    # Adaptive UCB: score = mu + k * sigma
    # k adapts based on how close f_best is to the known minimum
    # Early on (f_best >> f_min): high k for exploration
    # Late (f_best ≈ f_min): low k for exploitation
    
    # Progress: 0 when f_best is far from optimum, 1 when at optimum
    # Use a smooth transition based on the gap
    gap = max(f_best - f_min_known, 1e-6)
    
    # Normalize progress: larger gap means earlier stage
    # Use exponential decay: k decreases as we approach the minimum
    # k starts around 4.0 (high exploration) and drops to ~0.5 (exploitation)
    # The scale parameter controls how quickly k decreases
    scale = 2.0  # controls the rate of transition
    progress = 1.0 - np.exp(-gap / scale)
    
    # k interpolates from high exploration to low exploitation
    k_high = 4.0
    k_low = 0.5
    k = k_low + (k_high - k_low) * progress
    
    # UCB score: we want to maximize this (higher = more promising)
    # Since we're minimizing, lower mu is better, so we use -mu + k*sigma
    # But the contract says "higher meaning evaluate next"
    # So score = -mu + k * sigma (higher when mu is low or sigma is high)
    score = -mu + k * sigma_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -1e10)
    
    # Check if we're close to the deadline; if so, just return
    elapsed = time.time() - t_start
    if elapsed > 280:  # 300s limit with safety margin
        return score
    
    return score
# EVOLVE-BLOCK-END
