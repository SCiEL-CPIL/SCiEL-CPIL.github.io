import numpy as np
from scipy.stats import norm
import time

# Global time tracking for deadline guard
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using LCB (Lower Confidence Bound)
    for Bayesian optimization on Shekel 4d.
    
    LCB provides a more linear balance between exploration and exploitation
    than EI, which helps avoid getting trapped in local optima in high-dimensional
    multimodal landscapes.
    """
    global _start_time
    
    # Initialize time tracking
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we've used most of the 330s budget, return quickly
    elapsed = time.time() - _start_time
    if elapsed > 320:  # 10s safety margin
        # Return a simple, fast score to avoid timeout
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        poi = norm.cdf(z)
        score = poi / np.maximum(cost, 1e-9)
        score = np.where(np.isfinite(score), score, 0.0)
        return score
    
    # Ensure arrays are 1D and finite
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    cost = np.asarray(cost, dtype=np.float64).ravel()
    
    # Safety: replace non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Lower Confidence Bound (LCB)
    # LCB = mu - beta * sigma
    # For minimization, we want to minimize mu - beta*sigma
    # So the acquisition score (higher is better) is: -(mu - beta*sigma) = -mu + beta*sigma
    
    # Adaptive beta: start with exploration, shift to exploitation as budget depletes
    # Estimate progress based on sigma distribution (more evaluations -> lower sigma)
    sigma_mean = np.mean(sigma) if len(sigma) > 0 else 1.0
    sigma_max = np.max(sigma) if len(sigma) > 0 else 1.0
    
    # Beta starts high (exploration) and decreases as uncertainty drops
    # When sigma is high relative to max, we're early -> more exploration
    # When sigma is low, we're late -> more exploitation
    if sigma_max > 0:
        progress = 1.0 - (sigma_mean / sigma_max)  # 0 early, 1 late
    else:
        progress = 1.0
    
    # Beta from 2.0 (early) to 0.5 (late)
    beta = 2.0 - 1.5 * progress
    
    # LCB score (higher is better for minimization)
    lcb_score = -mu + beta * sigma
    
    # Normalize by cost (cost-aware)
    score = lcb_score / cost
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
