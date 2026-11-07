"""Rewritten program with UCB-based cost-aware acquisition."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB)
    with a linear cost penalty.
    
    Score = mu + beta * sigma - lambda * cost
    
    This balances exploration (high sigma), exploitation (low mu), 
    and cost efficiency (low cost).
    """
    # Ensure sigma is non-negative and handle potential numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # UCB parameters
    beta = 2.5      # Exploration constant
    lambda_cost = 0.5  # Cost penalty weight
    
    # Calculate UCB score
    ucb_score = mu + beta * sigma
    
    # Apply cost penalty
    final_score = ucb_score - lambda_cost * cost
    
    # Ensure finite scores
    final_score = np.nan_to_num(final_score, nan=0.0, posinf=1e9, neginf=-1e9)
    
    return final_score
# EVOLVE-BLOCK-END
