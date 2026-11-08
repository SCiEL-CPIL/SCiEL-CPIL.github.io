"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    with an exploration bonus (UCB-like) per unit cost.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions.
    sigma : array-like, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed function value.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : array, shape (2000,)
        Acquisition scores (higher = more promising).
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and ensure positivity
    sigma = np.maximum(sigma, 1e-9)
    
    # Clip cost to avoid division by zero
    cost = np.maximum(cost, 1e-9)
    
    # Compute z-score for expected improvement
    z = (f_best - mu) / sigma
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # UCB exploration term: balance exploitation with exploration
    # beta controls exploration vs exploitation trade-off
    # In 8D, we need more exploration to escape local optima
    beta = 0.5
    ucb = -mu + beta * sigma
    
    # Combine EI and UCB: EI captures expected gain, UCB captures uncertainty
    # Weight them to balance exploitation and exploration
    combined = ei + 0.3 * np.maximum(ucb, 0)
    
    # Cost-aware: normalize by cost
    scores = combined / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
