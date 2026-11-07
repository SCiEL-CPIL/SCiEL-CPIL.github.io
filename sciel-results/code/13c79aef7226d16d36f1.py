import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Standard Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Exploration bonus: UCB-like term
    # Small beta to not override EI
    beta = 0.5
    exploration_bonus = beta * sigma_safe
    
    # Combined score numerator
    numerator = ei + exploration_bonus
    
    # Softened cost penalty
    # Use sqrt(cost) to reduce the impact of cost differences
    # Cost is in [1, 2], sqrt is in [1, 1.414]
    # This makes expensive points less penalized
    cost_factor = np.sqrt(cost_safe)
    
    score = numerator / cost_factor
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
