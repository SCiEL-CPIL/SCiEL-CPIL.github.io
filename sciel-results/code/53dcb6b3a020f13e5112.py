# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses Gaussian Process Lower Confidence Bound (GP-LCB) for minimization.
    LCB = mu - beta * sigma
    This balances exploitation (mu) and exploration (sigma) without dividing by cost,
    which can cause numerical instability or misalignment in cost-aware settings.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units
    f_best : float
        Best observed function value so far
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    sigma = np.maximum(sigma, 1e-12)
    
    # GP-LCB for minimization:
    # We want to minimize the function. The lower confidence bound gives a lower estimate
    # of the function value. We prefer points with lower LCB.
    # However, the acquisition function should return HIGHER scores for better points.
    # So we return -LCB = -(mu - beta * sigma) = -mu + beta * sigma
    
    # Beta is the exploration/exploitation trade-off parameter.
    # A value around 2.0 is standard for LCB.
    beta = 2.0
    
    lcb = mu - beta * sigma
    score = -lcb
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
