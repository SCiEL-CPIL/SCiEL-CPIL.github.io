import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware acquisition with adaptive exploration
    # Use a time-aware approach: track elapsed time to avoid timeout
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-9)
    
    # Adaptive beta: scale exploration based on uncertainty and cost
    # Higher beta when sigma is large relative to improvement potential
    max_sigma = np.max(sigma)
    if max_sigma < 1e-6:
        max_sigma = 1e-6
    
    # Base beta that adapts to the scale of uncertainty
    base_beta = 2.0
    
    # Cost-aware: prefer cheaper points with high uncertainty
    # Use a penalty for high cost that increases with confidence in improvement
    cost_penalty = 0.5
    
    # Compute expected improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    from scipy.stats import norm
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware score: balance improvement potential against cost
    # Use a logarithmic cost penalty to avoid over-penalizing
    cost_factor = np.log1p(cost)  # log(1 + cost)
    
    # Primary score: EI per unit cost (with soft penalty)
    scores = ei / (1.0 + cost_penalty * cost_factor)
    
    # Add exploration bonus for high-uncertainty regions
    exploration_bonus = base_beta * sigma / (1.0 + cost_penalty * cost_factor)
    
    # Combine: exploit where EI is high, explore where sigma is high
    # Weight exploration more when we're early (high uncertainty overall)
    uncertainty_ratio = sigma / max_sigma
    exploration_weight = 0.3 + 0.7 * uncertainty_ratio
    
    final_scores = scores + exploration_weight * exploration_bonus
    
    # Ensure finite values
    final_scores = np.nan_to_num(final_scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return final_scores
