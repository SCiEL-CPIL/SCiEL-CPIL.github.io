"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Use UCB (Upper Confidence Bound) for minimization.
    # We want to maximize the score.
    # Lower predicted value (mu) is better. High uncertainty (sigma) is better.
    # UCB_value = mu - beta * sigma
    # We divide by cost to make it cost-aware.
    
    # Clip sigma to ensure it's non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Beta controls the exploration/exploitation trade-off.
    # 1.96 was used in inspiration (score 0.9957).
    # EI (current) is score 1.0.
    # Let's try a slightly more exploitative UCB, beta=1.5.
    beta = 1.5
    
    ucb_val = mu - beta * sigma
    
    # Cost-aware score: higher is better.
    # Since we minimize the objective, lower ucb_val is better.
    # So we negate it.
    scores = -ucb_val / cost
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
