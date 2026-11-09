"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using LCB with cost-efficiency weighting.
    
    mu, sigma, f_best, cost: arrays of shape (2000,)
    Returns: 2000 finite scores (higher = evaluate next)
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clamp sigma to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # LCB: mu - beta * sigma (lower = better for minimization)
    # We want high acquisition scores, so we negate the LCB
    # beta controls exploration; start moderate and let cost-weighting handle balance
    
    # Cost efficiency: prefer lower cost points
    # cost ranges from ~1 to ~2, so 1/cost ranges from ~0.5 to ~1.0
    # Use inverse cost as a weight
    cost_weight = 1.0 / np.maximum(cost, 1e-9)
    
    # Exploration coefficient: use a moderate beta
    # As f_best approaches 0, we should explore less
    # Scale beta by how far f_best is from the minimum (0)
    beta = 1.5
    
    # LCB score (higher is better for our purpose)
    lcb_score = -(mu - beta * sigma_safe)
    
    # Expected improvement component for exploitation
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Combine: weighted sum of LCB (exploration) and EI (exploitation)
    # Weight by cost efficiency
    # Use a blend: 0.6 * LCB + 0.4 * EI, both weighted by cost efficiency
    score = (0.6 * lcb_score + 0.4 * ei) * cost_weight
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
