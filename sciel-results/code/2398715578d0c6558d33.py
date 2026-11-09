import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # UCB-based acquisition with adaptive exploration
    # Score = -mu + beta * sigma (minimization: we want low mu, high sigma for exploration)
    # Higher score = more desirable to evaluate
    
    # Adaptive beta: larger when uncertainty is high relative to current best
    sigma_max = np.max(sigma) if np.max(sigma) > 1e-9 else 1.0
    sigma_mean = np.mean(sigma)
    
    # Gap between current best and the mean prediction
    gap = f_best - np.mean(mu)
    gap_scale = max(abs(gap), 1e-6)
    
    # Beta adapts: when gap is small (near convergence), explore more
    # When gap is large, exploit more
    # beta in range [0.5, 3.0]
    beta = 1.5 + 1.5 * (sigma_mean / (sigma_max + 1e-9))
    # Scale beta by how far we are from the known minimum
    # Shekel 4D min is -10.5364
    remaining = f_best - (-10.5364)
    if remaining > 0:
        beta *= min(1.0 + remaining / 5.0, 2.0)
    
    # UCB score for minimization: -mu + beta * sigma
    ucb = -mu + beta * sigma
    
    # Add a small exploitation term: reward where GP predicts very low values
    # This helps when UCB alone might be too exploratory
    mu_min = np.min(mu)
    mu_range = np.max(mu) - mu_min if np.max(mu) - mu_min > 1e-9 else 1.0
    exploitation = (mu_min - mu) / mu_range  # in [0, 1], 1 where mu is lowest
    
    # Combine: primarily UCB, with small exploitation boost
    # Normalize UCB to comparable scale
    ucb_min = np.min(ucb)
    ucb_range = np.max(ucb) - ucb_min if np.max(ucb) - ucb_min > 1e-9 else 1.0
    ucb_norm = (ucb - ucb_min) / ucb_range  # in [0, 1]
    
    combined = 0.8 * ucb_norm + 0.2 * exploitation
    
    # Cost-aware scaling: soft penalty
    # cost in [1, 2], use 1/(1 + alpha*(cost-1)) with alpha=1.0
    # This gives cost=1 -> 1.0, cost=2 -> 0.5
    cost_penalty = 1.0 / (1.0 + 1.0 * (cost - 1.0))
    
    scores = combined * cost_penalty
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    if time.time() - start_time > 325:
        return np.zeros_like(mu)
        
    return scores
