"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB) for minimization.
    
    Parameters:
    mu (np.array): Mean of the Gaussian process posterior at candidate points.
    sigma (np.array): Standard deviation of the Gaussian process posterior at candidate points.
    f_best (float): The best (lowest) observed value so far.
    
    Returns:
    np.array: Scores for each candidate point. Higher score indicates better candidate.
    """
    # Ensure sigma is positive
    sigma_safe = np.clip(sigma, 1e-8, None)
    
    # UCB for minimization:
    # We want to minimize f(x). The optimistic lower bound is mu - beta * sigma.
    # We want to select the point with the lowest lower bound.
    # Since the caller selects the point with the HIGHEST score,
    # we return the negative of the lower bound:
    # score = -(mu - beta * sigma) = -mu + beta * sigma
    
    # Beta controls exploration. 
    # A fixed value of 2.0 is a common heuristic for small budgets.
    beta = 2.0
    
    scores = -mu + beta * sigma_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
