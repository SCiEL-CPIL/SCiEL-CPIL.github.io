import numpy as np
import time
from scipy.stats import norm

# Global timer for deadline guard
_start_time = time.time()
_DEADLINE = 330.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    Uses Expected Improvement (EI) for minimization.
    
    Parameters:
    mu (np.ndarray): Posterior mean, shape (2000,).
    sigma (np.ndarray): Posterior standard deviation, shape (2000,).
    f_best (float): Best observed value so far.
    
    Returns:
    np.ndarray: Scores for the 2000 candidates. Higher is better.
    """
    # Deadline guard
    if time.time() - _start_time > _DEADLINE - 5.0:
        # Fallback to pure exploitation to save time
        scores = -mu
        scores = np.where(np.isfinite(scores), scores, -1e6)
        return scores

    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-10)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-10)
    mu = np.where(np.isfinite(mu), mu, f_best)

    # Expected Improvement for minimization
    # EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x))/sigma(x)) + sigma(x) * phi((f_best - mu(x))/sigma(x))
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # EI is non-negative, higher is better
    # Ensure finite scores
    scores = np.where(np.isfinite(ei), ei, 0.0)
    
    return scores
