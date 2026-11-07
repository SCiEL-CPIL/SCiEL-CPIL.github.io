import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of hartmann 6d.
    Uses a hybrid approach: Expected Improvement (EI) with a small exploration
    parameter, combined with a UCB-like boost for high-uncertainty regions.
    
    Parameters:
    -----------
    mu : array of shape (N,)
        Mean of the Gaussian Process posterior at N candidate points.
    sigma : array of shape (N,)
        Standard deviation of the Gaussian Process posterior at N candidate points.
    f_best : float
        The lowest observed function value so far.
        
    Returns:
    --------
    scores : array of shape (N,)
        Higher scores indicate points with higher potential for improvement (lower true function value).
    """
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Calculate the z-score for the expected improvement
    # Using a small xi to balance exploitation and exploration
    xi = 0.01
    z = (f_best - mu - xi) / sigma_safe
    
    # Compute Expected Improvement
    # EI = (f_best - mu - xi) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a UCB-like exploration term to encourage exploring high-uncertainty regions
    # This helps escape local minima in the 6D space
    # The coefficient is chosen to be small so that EI dominates in low-uncertainty regions
    ucb_boost = 0.1 * sigma_safe
    
    # Combine EI and UCB boost
    scores = ei + ucb_boost
    
    # Ensure all returned scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
