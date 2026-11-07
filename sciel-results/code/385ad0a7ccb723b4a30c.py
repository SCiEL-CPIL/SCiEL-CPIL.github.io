"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB).
    
    Parameters:
    -----------
    mu : array-like, shape (n,)
        Mean predictions from the GP.
    sigma : array-like, shape (n,)
        Standard deviation predictions from the GP.
    f_best : float
        Best observed value so far.
    cost : array-like, shape (n,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores (higher is better).
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # UCB Parameter (beta)
    # Controls the trade-off between exploitation (-mu) and exploration (sigma)
    # Since we are minimizing, we maximize -mu + beta * sigma
    beta = 2.5
    
    # Cost Penalty Exponent
    # Higher gamma penalizes high-cost points more
    gamma = 1.2
    
    # Calculate UCB term
    # We want low mu (good prediction) and high sigma (uncertainty)
    ucb_term = -mu + beta * sigma_safe
    
    # Cost-aware score
    scores = ucb_term / (cost_safe ** gamma)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
