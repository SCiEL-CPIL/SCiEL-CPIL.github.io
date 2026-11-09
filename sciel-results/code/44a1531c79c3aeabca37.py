# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """Cost-aware acquisition function combining EI and UCB with exploration bonus."""
    # Safety: ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # UCB for exploration
    beta = 2.0  # exploration parameter
    ucb = mu + beta * sigma_safe
    
    # Cost-aware weighting: prefer cheaper points
    # Normalize cost to [0, 1] range for stable weighting
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    if cost_range > 1e-12:
        cost_norm = (cost - cost_min) / cost_range
    else:
        cost_norm = np.ones_like(cost) * 0.5
    
    # Cost discount: exponential decay based on normalized cost
    # Cheaper points get higher weight
    cost_weight = np.exp(-0.5 * cost_norm)
    
    # Combine EI and UCB with cost awareness
    # Weight EI more heavily for exploitation, UCB for exploration
    # Scale EI by cost weight, add UCB component
    score_ei = ei * cost_weight
    score_ucb = -ucb * cost_weight  # Negate because we want to minimize
    
    # Blend: 70% EI, 30% UCB
    score = 0.7 * score_ei + 0.3 * score_ucb
    
    # Add small exploration bonus for high-uncertainty regions
    # This helps escape local minima
    exploration_bonus = 0.1 * sigma_safe * cost_weight
    
    score = score + exploration_bonus
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
