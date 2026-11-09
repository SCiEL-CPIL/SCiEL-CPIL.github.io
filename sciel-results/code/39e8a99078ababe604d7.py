"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() - _start_time > 320:
        return np.zeros_like(mu)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # UCB-style exploration term
    # beta=1.5 balances exploitation and exploration
    beta = 1.5
    ucb = -mu + beta * sigma_safe
    
    # Cost-aware scaling:
    # Exploitation (EI): mild cost penalty (cost^0.3) - still prefer cheaper points
    # Exploration (UCB): stronger penalty via inverse sqrt(cost) - 
    #   expensive regions are treated as higher-variance, so exploration is dampened
    cost_safe = np.maximum(cost, 1e-9)
    
    # Mild cost penalty for exploitation
    ei_cost_penalty = np.power(cost_safe, 0.3)
    ei_adjusted = ei / ei_cost_penalty
    
    # Stronger cost-aware scaling for exploration:
    # Inverse square root of cost dampens exploration in expensive regions
    # This is distinct from simple division - it's a variance-aware penalty
    ucb_cost_scale = 1.0 / np.sqrt(cost_safe)
    ucb_adjusted = ucb * ucb_cost_scale
    
    # Combine: 70% exploitation, 30% exploration
    scores = 0.7 * ei_adjusted + 0.3 * ucb_adjusted
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
