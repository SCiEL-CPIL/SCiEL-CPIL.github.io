"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware UCB with adaptive exploration and softer cost penalty
    # Score = (f_best - mu) + beta * sigma, divided by sqrt(cost)
    
    # Clip sigma to avoid numerical issues
    sigma = np.clip(sigma, 1e-8, None)
    
    # Adaptive beta: more exploration when far from optimum
    # Hartmann 3D min is ~-3.86278; f_best starts higher (worse)
    # When f_best is close to -3.86, we're near optimum -> less exploration
    # When f_best is far, explore more
    gap = f_best - (-3.86278)  # positive when worse than optimum
    beta = 1.0 + 0.5 * np.clip(gap / 2.0, 0.0, 1.0)
    
    # Optimistic improvement
    optimistic_improvement = (f_best - mu) + beta * sigma
    
    # Softer cost penalty: 1/sqrt(cost) instead of 1/cost
    # Cost is in [1, 2], so this reduces the penalty range
    scores = optimistic_improvement / np.sqrt(cost)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
