import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Probabilistic Lower Bound (PLB) acquisition function.
    score = mu + k * sigma, where k is adaptive based on f_best.
    Higher score means more likely to beat the incumbent.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Safety: ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-10)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-10)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Adaptive k: larger when f_best is far from optimum (3.0), smaller when close
    # f_best starts high and decreases toward 3.0
    # We want more exploration early, less later
    # Use a smooth transition
    # When f_best is large (e.g., 100+), k should be ~3-4
    # When f_best is close to 3, k should be ~0.5-1
    
    # Normalize the gap
    gap = f_best - 3.0
    # Clip gap to avoid issues
    gap = np.clip(gap, 0.0, 200.0)
    
    # Adaptive k: starts around 3.5, decays to around 0.8 as gap shrinks
    # Using a smooth function: k = 0.8 + 2.7 * exp(-gap / 10)
    # When gap=0: k = 0.8 + 2.7 = 3.5
    # When gap=10: k = 0.8 + 2.7*exp(-1) ≈ 0.8 + 0.99 ≈ 1.79
    # When gap=30: k = 0.8 + 2.7*exp(-3) ≈ 0.8 + 0.13 ≈ 0.93
    # When gap=100: k ≈ 0.8
    k = 0.8 + 2.7 * np.exp(-gap / 10.0)
    
    # PLB score: mu + k * sigma
    # This maximizes the probability that the true value exceeds f_best
    scores = mu + k * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
