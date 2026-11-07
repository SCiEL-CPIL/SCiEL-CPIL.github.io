import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 300.0  # seconds, with safety margin

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement with a small UCB
    exploration boost for Bayesian optimization of hartmann 6d.
    
    The EI term drives exploitation of regions likely to improve upon f_best,
    while the UCB term (0.1 * sigma) encourages exploration of high-uncertainty
    regions, helping escape local minima in the 6D space.
    
    Parameters:
    -----------
    mu : numpy array of shape (N,)
        Posterior mean of the GP at N candidate points.
    sigma : numpy array of shape (N,)
        Posterior standard deviation of the GP at N candidate points.
    f_best : float
        The best (lowest) function value observed so far.
        
    Returns:
    --------
    scores : numpy array of shape (N,)
        Combined EI + UCB scores. Higher is better.
    """
    # Time guard: if we're close to the deadline, return a simple fallback
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT - 5.0:
        # Fallback: return sigma-based scores to still be valid
        sigma_safe = np.maximum(sigma, 1e-6)
        scores = sigma_safe
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Small exploration parameter to balance exploitation and exploration
    xi = 0.01
    
    # Calculate the z-score for the expected improvement
    z = (f_best - mu - xi) / sigma_safe
    
    # Compute Expected Improvement
    # EI = (f_best - mu - xi) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a UCB-like exploration term to encourage exploring high-uncertainty regions
    # The coefficient 0.1 is small enough that EI dominates in low-uncertainty regions
    ucb_boost = 0.1 * sigma_safe
    
    # Combine EI and UCB boost
    scores = ei + ucb_boost
    
    # Ensure all returned scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
