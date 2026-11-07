import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Diversity-promoting UCB strategy
    # Uses a weighted sum of exploitation (f_best - mu) and exploration (sigma)
    # High sigma (unexplored regions) is explicitly rewarded to promote diversity
    # and penalize points close to already-evaluated locations (low sigma).
    
    # Exploration weight. Higher beta encourages more exploration.
    beta = 3.0
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0.0)
    
    # Prevent division by zero
    cost_safe = np.maximum(cost, 1e-8)
    
    # UCB-like score: (f_best - mu) + beta * sigma
    # We maximize this score.
    scores = (f_best - mu + beta * sigma) / cost_safe
    
    return scores
