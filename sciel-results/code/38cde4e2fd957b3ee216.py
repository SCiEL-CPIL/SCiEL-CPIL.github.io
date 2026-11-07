import numpy as np
import time

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    deadline = start_time + 290.0  # 40s safety margin from 330s limit
    
    # Hypervolume-based acquisition function
    # We treat (mu, sigma) as a bi-objective problem where we want to minimize both
    # The idea is to score candidates by the hypervolume they would add to the Pareto front
    
    # Normalize mu and sigma to [0, 1] range for consistent scaling
    mu_min, mu_max = np.min(mu), np.max(mu)
    sigma_min, sigma_max = np.min(sigma), np.max(sigma)
    
    mu_range = mu_max - mu_min
    sigma_range = sigma_max - sigma_min
    
    if mu_range < 1e-10:
        mu_range = 1.0
    if sigma_range < 1e-10:
        sigma_range = 1.0
    
    mu_norm = (mu - mu_min) / mu_range
    sigma_norm = (sigma - sigma_min) / sigma_range
    
    # Compute a combined score that favors points with low mu and low sigma
    # but with a non-linear trade-off to encourage diversity
    # Using a power-based weighting to create a more interesting Pareto front
    power_mu = 0.7
    power_sigma = 0.3
    
    # Score is based on the "dominance" in the normalized space
    # Points that are better in both objectives get higher scores
    # We use a weighted geometric mean to balance exploration and exploitation
    base_score = (1 - mu_norm) ** power_mu * (1 - sigma_norm) ** power_sigma
    
    # Add a small perturbation based on the position to break ties
    # and encourage exploration of diverse regions
    diversity_bonus = 0.1 * np.sin(np.pi * mu_norm) * np.sin(np.pi * sigma_norm)
    
    scores = base_score + diversity_bonus
    
    # Ensure all scores are finite and positive
    scores = np.where(np.isfinite(scores) & (scores > 0), scores, 1e-10)
    
    # Check deadline
    if time.time() > deadline:
        return scores
        
    return scores
