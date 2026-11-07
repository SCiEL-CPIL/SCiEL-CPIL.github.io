"""Rewritten program using GP-LCB acquisition for cost-aware Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware GP-LCB acquisition function.
    
    Uses a Lower Confidence Bound (LCB) objective: mu - beta * sigma.
    Maximizes (beta * sigma - mu) / cost.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed value so far
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = better)
    """
    # Beta = 2.0 encourages more exploration to escape local minima in Ackley.
    beta = 2.0
    
    # LCB score (higher is better): maximize beta * sigma - mu
    base_score = beta * sigma - mu
    
    # Cost-aware penalty: divide by cost.
    # This penalizes expensive evaluations more strongly than 1/sqrt(cost),
    # directing search towards cheaper regions unless potential is very high.
    scores = base_score / (cost + 1e-9)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
