"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function for Bayesian optimization.
    
    Samples a single realization from the Gaussian posterior and returns
    the negative sampled values as scores. Lower sampled values (better
    objective estimates) get higher scores.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Posterior mean at candidate points
    sigma : array-like, shape (2000,)
        Posterior standard deviation at candidate points
    f_best : float
        Best observed value so far
        
    Returns:
    --------
    scores : array, shape (2000,)
        Thompson sampling scores (higher is better)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid sampling issues
    sigma_safe = np.where(sigma > 1e-10, sigma, 1e-10)
    
    # Sample from the posterior: f(x) ~ N(mu(x), sigma(x)^2)
    # We want to minimize f, so we sample and prefer lower values
    rng = np.random.default_rng()
    samples = mu + sigma_safe * rng.standard_normal(mu.shape)
    
    # For minimization, we want to maximize the score
    # So we return negative samples (lower f -> higher score)
    scores = -samples
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
