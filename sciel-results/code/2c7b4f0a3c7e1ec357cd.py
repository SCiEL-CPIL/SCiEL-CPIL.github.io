import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        Posterior mean predictions
    sigma : np.ndarray, shape (2000,)
        Posterior standard deviation predictions
    f_best : float
        Best (lowest) function value observed so far
    
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Acquisition scores (higher is better for selection)
    """
    # Use Expected Improvement (EI) with adaptive exploration
    # EI is theoretically grounded and handles multimodal functions well
    
    # Ensure sigma is positive for numerical stability
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute Z-score: how many sigma below f_best the mean is
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement: E[f_best - f(x)]
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is standard normal CDF and phi is standard normal PDF
    
    # Compute EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a small UCB-like exploration term to encourage diversity
    # This helps escape local optima in the multimodal Michalewicz function
    # Use a small weight for the exploration term
    exploration_weight = 0.1
    ucb_term = exploration_weight * sigma_safe
    
    # Combine EI and exploration
    scores = ei + ucb_term
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure scores are non-negative (EI can be slightly negative due to numerical issues)
    scores = np.maximum(scores, 0.0)
    
    return scores
