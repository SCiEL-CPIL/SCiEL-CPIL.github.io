"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Lower Confidence Bound (LCB) for minimization.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (n,)
        Mean of the GP posterior at candidate points.
    sigma : np.ndarray, shape (n,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed value so far (lowest, since we're minimizing).
    
    Returns:
    --------
    scores : np.ndarray, shape (n,)
        Acquisition scores; higher means more desirable to evaluate.
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    
    # Replace non-finite values
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # LCB for minimization: lower bound on the objective value
    # LCB = mu - beta * sigma
    # Higher scores mean more desirable (lower LCB = potentially better minimum)
    # So we return -LCB = -mu + beta * sigma to make higher = better
    beta = 1.5
    lcb = mu - beta * sigma
    scores = -lcb  # = -mu + beta * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
