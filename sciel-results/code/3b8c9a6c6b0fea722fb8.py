"""Rewritten program using Cost-Weighted Lower Confidence Bound (CW-LCB)."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Weighted Lower Confidence Bound (CW-LCB) acquisition function.
    
    This approach uses the Lower Confidence Bound (LCB) for minimization,
    which balances exploitation (low mean) and exploration (high variance).
    The score is defined as the potential improvement (f_best - mu + beta * sigma)
    divided by the evaluation cost.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        Best observed value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : array, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma and cost are positive to avoid division by zero
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Beta parameter for the confidence bound
    # A value of 2.0 is a common heuristic for balancing exploration and exploitation
    beta = 2.0
    
    # Calculate the Lower Confidence Bound potential improvement
    # We want to minimize mu - beta * sigma, which is equivalent to maximizing
    # (f_best - mu + beta * sigma) relative to f_best.
    # If mu is much lower than f_best, this term is large.
    # If sigma is large, this term is larger (exploration).
    lcb_improvement = f_best - mu + beta * sigma_safe
    
    # Clip negative improvements to 0 to focus on points with potential to beat f_best
    lcb_improvement = np.maximum(lcb_improvement, 0.0)
    
    # Cost-weighted score
    scores = lcb_improvement / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
