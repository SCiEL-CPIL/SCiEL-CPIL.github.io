"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function based on probability-weighted improvement.
    
    Args:
        mu: array of mean predictions, shape (2000,)
        sigma: array of standard deviations, shape (2000,)
        f_best: scalar, lowest observed value
        cost: array of costs, shape (2000,)
        
    Returns:
        scores: array of acquisition scores, shape (2000,)
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma
    
    # Probability of improvement P(mu < f_best)
    # Using CDF of standard normal
    prob = norm.cdf(z)
    
    # Deterministic improvement magnitude
    # If mu < f_best, this is positive.
    imp = f_best - mu
    
    # Score: Probability-weighted improvement per unit cost
    # This aligns with the stochastic knapsack heuristic suggested
    scores = (prob * imp) / cost
    
    # Handle any numerical issues
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
