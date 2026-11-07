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
    # Clip sigma to avoid division by zero or numerical issues, though UCB doesn't divide
    sigma_safe = np.clip(sigma, 1e-8, None)
    
    # Exploration parameter beta
    # A value of 2.0 is a common heuristic for balancing exploration and exploitation
    beta = 2.0
    
    # UCB for minimization: we want to minimize mu - beta * sigma
    # Or maximize -mu + beta * sigma
    # We return scores where higher is better
    score = -mu + beta * sigma_safe
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
