"""Rewritten cost-aware acquisition: Cost-Effective Expected Improvement (CEI)."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    # mu, sigma, cost: shape (2000,). Return 2000 finite scores (higher = evaluate next).
    # Approach: Cost-Aware Upper Confidence Bound (CA-UCB)
    # For minimization, we use the Lower Confidence Bound (LCB) as an optimistic estimate.
    # LCB(x) = mu(x) - beta * sigma(x)
    # We want to minimize LCB(x) while accounting for cost.
    # Score = -LCB(x) / cost(x)
    
    sigma = np.maximum(sigma, 1e-9)
    
    # Hyperparameter for exploration/exploitation balance
    # Beta controls the weight of uncertainty.
    # A value of 1.5 is a reasonable default for moderate exploration.
    beta = 1.5
    
    # Calculate Lower Confidence Bound (optimistic value for minimization)
    lcb = mu - beta * sigma
    
    # We want to minimize LCB, so we maximize -LCB.
    # Normalize by cost to prefer cheaper evaluations.
    score = -lcb / cost
    
    # Safety: keep everything finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e12, neginf=-1e12)
    
    return score
# EVOLVE-BLOCK-END
