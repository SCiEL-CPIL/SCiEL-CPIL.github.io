"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Combines Expected Improvement (EI) per unit cost with an uncertainty
    bonus (UCB) to balance exploitation and exploration under cost constraints.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions.
    sigma : array-like, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed function value.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = better candidate to evaluate next).
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: ensure no NaNs or infs in inputs
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-9, posinf=1e-9, neginf=1e-9)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1.0, neginf=1.0)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-6)
    
    # Compute Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # EI per unit cost
    ei_cost = ei / cost
    
    # Uncertainty bonus (UCB-style) scaled by cost
    # Use a temperature parameter to balance exploration
    # The UCB term encourages exploration in high-uncertainty regions
    # but penalizes expensive points
    ucb = sigma / cost
    
    # Combine EI and UCB with a weighted sum
    # The weight on UCB can be tuned; here we use a moderate value
    # to encourage some exploration without over-emphasizing uncertainty
    alpha = 0.5
    scores = ei_cost + alpha * ucb
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
