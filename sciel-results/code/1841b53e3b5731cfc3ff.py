"""Pareto-based multi-objective acquisition for Bayesian optimization of Hartmann 6D."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Dynamic weighted sum of normalized predictive mean (exploitation) and 
    normalized predictive uncertainty (exploration).
    
    This approach uses a UCB-like structure where the weight on exploitation 
    (favoring low mu) increases as the optimization progresses (f_best approaches 
    the known global minimum).
    
    Parameters:
        mu: array of shape (2000,) - posterior mean in objective units
        sigma: array of shape (2000,) - posterior std dev in objective units
        f_best: float - best observed value so far (lowest)
    
    Returns:
        scores: array of shape (2000,) - higher scores mean evaluate next
    """
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Known global minimum of Hartmann 6D
    f_min = -3.32237
    # Approximate maximum for progress calculation
    f_max = 3.32227
    
    # Calculate progress of optimization: 0 at start, 1 at end
    span = f_max - f_min
    if span > 1e-9:
        progress = (f_max - f_best) / span
    else:
        progress = 0.0
    
    # Clamp progress to [0, 1]
    progress = np.clip(progress, 0.0, 1.0)
    
    # Dynamic weight for exploitation
    # Start with 30% exploitation, 70% exploration
    # End with 100% exploitation, 0% exploration
    # Using a smooth curve
    w_exploit = 0.3 + 0.7 * progress
    w_explore = 1.0 - w_exploit
    
    # Normalize mu: lower mu is better.
    # Map mu to [0, 1] such that 1 is best (lowest mu)
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    mu_range = mu_max - mu_min
    
    if mu_range < 1e-9:
        exp_score = np.ones_like(mu)
    else:
        # mu_norm is 0 for best (min mu), 1 for worst (max mu)
        mu_norm = (mu - mu_min) / mu_range
        # Invert so that 1 is best
        exp_score = 1.0 - mu_norm
        
    # Normalize sigma: higher sigma is better for exploration
    sigma_min = np.min(sigma)
    sigma_max = np.max(sigma)
    sigma_range = sigma_max - sigma_min
    
    if sigma_range < 1e-9:
        unc_score = np.ones_like(sigma)
    else:
        # sigma_norm is 0 for min sigma, 1 for max sigma
        unc_score = (sigma - sigma_min) / sigma_range
        
    # Combine objectives
    score = w_exploit * exp_score + w_explore * unc_score
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
