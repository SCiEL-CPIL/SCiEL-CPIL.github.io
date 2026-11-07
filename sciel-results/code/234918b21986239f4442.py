"""Rewritten cost-aware acquisition: Cost-Effective Upper Confidence Bound (CE-UCB)."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    # mu, sigma, cost: shape (2000,). Return 2000 finite scores (higher = evaluate next).
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Upper Confidence Bound for minimization:
    # We want to maximize the lower bound of the objective function.
    # Lower bound = mu - beta * sigma.
    # To maximize the lower bound, we maximize (beta * sigma - mu).
    # This balances exploitation (low mu) and exploration (high sigma).
    
    # Exploration parameter beta
    # A fixed beta of 1.0 is a standard heuristic.
    beta = 1.0
    
    # UCB score component
    ucb = beta * sigma - mu
    
    # Cost-Aware Acquisition: Cost-Effective UCB
    # We prefer high UCB per unit cost.
    score = ucb / cost
    
    # Safety: keep everything finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e12, neginf=-1e12)
    
    return score
# EVOLVE-BLOCK-END
