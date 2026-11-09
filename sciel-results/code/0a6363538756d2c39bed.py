"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed function value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = better to evaluate next).
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma = np.maximum(sigma, 1e-10)
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware adjustment:
    # The key insight is that the global minimum is in the intermediate cost region,
    # not at the cheap corners. We need to soften the cost penalty to allow exploration
    # of intermediate-cost regions.
    # 
    # Instead of dividing by cost (which over-penalizes intermediate costs), we use
    # a softer penalty: cost^alpha where alpha < 1, or a linear penalty with a small weight.
    # 
    # A good approach: EI / (1 + beta * cost) where beta is small, or
    # EI * (1 / (cost^gamma)) with gamma < 1.
    # 
    # From the measured finding: softening cost penalty improved score from 1.0 to 1.255.
    # Let's use EI / (cost^0.5) which is softer than EI / cost.
    
    # Also add a small exploration bonus for high uncertainty regions
    # to encourage exploration in early iterations when f_best is not yet good.
    
    # Soft cost penalty: use cost^0.5 instead of cost
    cost_penalty = np.sqrt(np.maximum(cost, 1e-10))
    
    scores = ei / cost_penalty
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
