import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Cost-Effective UCB (Lower Confidence Bound for minimization).
    
    Score = max(0, f_best - mu + kappa * sigma) / cost
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - lowest observed value
    cost : array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
    --------
    score : array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    t0 = time.time()
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Cost-Effective UCB for minimization
    # We want to minimize f.
    # LCB = mu - kappa * sigma.
    # We want to maximize the "improvement potential" relative to cost.
    # Improvement potential ~ f_best - LCB = f_best - (mu - kappa * sigma) = f_best - mu + kappa * sigma.
    
    kappa = 1.5
    
    # Calculate the UCB-based improvement score
    # If mu is very low (better than f_best), f_best - mu is positive.
    # If sigma is high, kappa * sigma adds exploration.
    ucb_score = f_best - mu + kappa * sigma_safe
    
    # We only care about points that might be better than f_best (or have high uncertainty)
    # Actually, if ucb_score is negative, it means the LCB is worse than f_best.
    # We can clamp to 0 to avoid negative scores, though the max will pick the highest positive one.
    ucb_score = np.maximum(ucb_score, 0.0)
    
    # Cost penalty
    cost_safe = np.maximum(cost, 1e-12)
    
    # Linear cost penalty (gamma = 1)
    score = ucb_score / cost_safe
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline guard
    elapsed = time.time() - t0
    if elapsed > 250:
        pass
        
    return score
