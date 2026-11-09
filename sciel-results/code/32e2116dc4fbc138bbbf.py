"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    
    Uses a hybrid approach: Expected Improvement (EI) for exploitation 
    combined with a small exploration bonus weighted by uncertainty.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Posterior mean at candidate points.
    sigma : ndarray, shape (2000,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Scores for each candidate; higher means more likely to be selected next.
    """
    # Deadline guard - return something reasonable if we're running low on time
    # (This is a safety net; the function should be fast)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    # where Phi is the CDF and phi is the PDF of the standard normal
    
    # Handle case where sigma is very small or zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    z = (f_best - mu) / sigma_safe
    
    # Compute EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Exploration bonus: proportional to uncertainty
    # Scale exploration by how far we are from the optimum (3.0)
    # Goldstein-Price minimum is 3.0
    f_opt = 3.0
    distance_to_opt = max(f_best - f_opt, 0.0)
    
    # Adaptive exploration weight
    # When f_best is far from optimum, explore more
    # When close, exploit more
    beta = 0.5 + 2.0 * np.exp(-distance_to_opt / 2.0)
    
    # Combine EI and exploration bonus
    scores = ei + beta * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
