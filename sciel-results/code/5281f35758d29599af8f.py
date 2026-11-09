"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of styblinski tang 1d.
    
    Uses Lower Confidence Bound (LCB) for minimization.
    LCB provides a more aggressive exploration strategy than EI, which is
    beneficial for multimodal functions like Styblinski-Tang where the
    global minimum is separated from local minima by significant barriers.
    By maximizing (beta * sigma - mu), we prioritize regions with high
    uncertainty and low predicted mean.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : array_like, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed value so far (minimum).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores for each candidate. Higher is better.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is strictly positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Compute Lower Confidence Bound (LCB) for minimization.
    # We want to maximize the acquisition score.
    # LCB value = mu - beta * sigma.
    # Since we minimize, a lower LCB is better.
    # Therefore, we return -LCB = beta * sigma - mu.
    
    # Beta controls the exploration-exploitation trade-off.
    # A value of 1.0 is a standard starting point for standardized objectives.
    beta = 1.0
    
    scores = beta * sigma - mu
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
