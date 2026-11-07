"""Cost-aware acquisition function for Bayesian optimization (Griewank 2D)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid of Expected Improvement
    and a UCB-style exploration bonus, divided by cost.
    
    Parameters:
    mu : array-like, shape (2000,)
        Mean of the GP posterior.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        The lowest observed function value so far.
    cost : array-like, shape (2000,)
        The cost of evaluating each candidate point.
        
    Returns:
    scores : array, shape (2000,)
        Acquisition scores. Higher scores indicate more desirable points.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # UCB exploration bonus: encourages visiting regions with high uncertainty
    # The coefficient 1.0 balances exploitation (EI) and exploration (sigma)
    ucb_bonus = sigma_safe
    
    # Hybrid acquisition: EI + UCB bonus
    # This is equivalent to a UCB with kappa=1 applied to the negative objective
    # but combined with EI for better exploitation
    hybrid = ei + ucb_bonus
    
    # Cost-aware: divide by cost
    scores = hybrid / cost_safe
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
