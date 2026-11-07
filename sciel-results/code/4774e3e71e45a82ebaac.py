"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 325.0


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    Uses Lower Confidence Bound (LCB) for minimization.
    Maximizes: -mu + beta * sigma
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the Gaussian process posterior.
    sigma : array-like, shape (2000,)
        Standard deviation of the Gaussian process posterior.
    f_best : float
        Best observed value so far (minimum).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores, higher means more promising.
    """
    # Deadline guard
    if time.time() - _start_time > _DEADLINE:
        return -mu + 2.5 * np.maximum(sigma, 1e-9)
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Avoid division by zero or negative sigma (though sigma should be positive)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # LCB for minimization: we want to minimize f(x).
    # The lower bound of the true value is mu - beta * sigma.
    # We want to find the point with the lowest lower bound.
    # So we maximize the acquisition function: -mu + beta * sigma.
    
    beta = 2.5
    
    scores = -mu + beta * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
