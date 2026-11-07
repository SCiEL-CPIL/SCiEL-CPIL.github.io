"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid of Expected Improvement (EI)
    and Uncertainty (UCB) to balance exploitation and exploration,
    normalized by the evaluation cost.
    
    Args:
        mu (np.ndarray): Mean of the GP posterior, shape (2000,).
        sigma (np.ndarray): Standard deviation of the GP posterior, shape (2000,).
        f_best (float): The lowest observed value so far.
        cost (np.ndarray): Cost of evaluating each candidate, shape (2000,).
        
    Returns:
        np.ndarray: Acquisition scores, shape (2000,). Higher is better.
    """
    # Ensure sigma is positive and not too small for numerical stability
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Calculate Uncertainty term (part of UCB)
    # We add a scaled uncertainty to encourage exploration
    # A fixed beta is used as we don't have the step count t
    beta = 2.0 
    ucb_term = beta * sigma
    
    # Hybrid score: EI + UCB term
    # This combines the expected gain with the potential for discovery
    combined_score = ei + ucb_term
    
    # Cost-aware normalization
    # We divide by cost to prefer cheaper evaluations
    # Using cost directly (gamma=1)
    # We add a small epsilon to cost to avoid division by zero (though cost >= 1)
    scores = combined_score / (cost + 1e-9)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
