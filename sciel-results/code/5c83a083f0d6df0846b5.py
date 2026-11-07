"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Expected improvement per unit cost, plus an exploration bonus.
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Score = EI / cost + lambda * sigma / cost
    
    # Safety margin for sigma to avoid division by zero
    sigma_safe = sigma + 1e-9
    
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware score
    # Ensure cost is positive (it is, min 1)
    cost_safe = cost + 1e-9
    
    # Exploration weight
    # 0.05 was a successful baseline. Let's try 0.1 for stronger exploration
    # in the early stages of the budget.
    lambda_exploration = 0.1
    
    score = ei / cost_safe + lambda_exploration * sigma / cost_safe
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
