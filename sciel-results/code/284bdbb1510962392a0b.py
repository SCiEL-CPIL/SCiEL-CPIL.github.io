"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-UCB (Upper Confidence Bound) for minimization.
    # We want to minimize mu - beta * sigma.
    # The acquisition score to maximize is -(mu - beta * sigma) / cost.
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Beta controls exploration. 
    # A fixed beta is a structural change from the implicit beta in CEI.
    # 1.5 is a reasonable default for exploration in BO.
    beta = 1.5
    
    # UCB value (lower is better for minimization)
    ucb_values = mu - beta * sigma
    
    # Cost-aware score (higher is better)
    # We divide by cost to penalize expensive points.
    # We negate the UCB value because we want to MAXIMIZE the score, 
    # but UCB minimizes the value.
    scores = -ucb_values / cost
    
    # Handle any non-finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
