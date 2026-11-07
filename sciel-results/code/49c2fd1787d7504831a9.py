"""Rewritten program using Cost-Adjusted Upper Confidence Bound (C-UCB)."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Adjusted Upper Confidence Bound (C-UCB).
    
    This approach uses the Upper Confidence Bound (UCB) strategy, which balances
    exploitation (low mu) and exploration (high sigma). For minimization, we
    want to find points where the lower bound of the posterior (mu - beta * sigma)
    is significantly lower than the current best f_best.
    
    The score is defined as the optimistic improvement over the current best,
    normalized by the evaluation cost:
    
    Score = (f_best - mu + beta * sigma) / cost
    
    Higher scores indicate candidates that are likely to be much better than
    f_best, adjusted for the cost of evaluation.
    """
    # Exploration-exploitation trade-off parameter
    beta = 2.0
    
    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 0.0)
    
    # Calculate the optimistic bound relative to f_best
    # We want high scores for low mu and high sigma
    optimistic_improvement = f_best - mu + beta * sigma_safe
    
    # Cost-aware weighting
    # Divide by cost to penalize expensive evaluations
    # Cost is at least 1.0
    score = optimistic_improvement / cost
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
