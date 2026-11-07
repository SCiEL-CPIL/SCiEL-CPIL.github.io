"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Probability of Improvement (PI).
    
    PI encourages points that have a high probability of improving upon the current best.
    Unlike Expected Improvement (EI), which weights by the magnitude of the expected improvement,
    PI treats all potential improvements equally in terms of probability mass.
    This can be beneficial for multimodal functions like Ackley, where finding *any*
    improvement is more important than maximizing the expected gain in a single step,
    especially in the early stages of search.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed value so far
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = more promising)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute z-score
    # z = (f_best - mu) / sigma
    # If z is large positive, P(f(x) < f_best) is high (close to 1)
    # If z is large negative, P(f(x) < f_best) is low (close to 0)
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    # CDF of the standard normal distribution
    pi_scores = norm.cdf(z)
    
    # Ensure finite values
    pi_scores = np.where(np.isfinite(pi_scores), pi_scores, 0.0)
    
    return pi_scores
# EVOLVE-BLOCK-END
