"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI)
    with a time-bounded deadline guard.
    
    Parameters:
    mu: array of shape (2000,) - GP mean predictions
    sigma: array of shape (2000,) - GP standard deviation predictions
    f_best: float - lowest observed objective value
    cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    scores: array of shape (2000,) - acquisition scores (higher is better)
    """
    t_start = time.time()
    DEADLINE = 300.0  # Safety margin before 330s limit
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero or negative sigma
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement for minimization:
    # EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    # where z = (f_best - mu(x)) / sigma(x)
    # Higher EI means more expected improvement over f_best.
    
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues in norm.cdf/pdf
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware: divide by cost to prefer cheaper evaluations
    # This makes the acquisition function cost-sensitive
    scores = ei / cost
    
    # Handle any non-finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Check if we're approaching the deadline
    elapsed = time.time() - t_start
    if elapsed > DEADLINE - 5.0:
        # Return a simple fallback to avoid timeout
        # Prefer points with low mu and low cost
        fallback = -mu / cost
        fallback = np.where(np.isfinite(fallback), fallback, -np.inf)
        return fallback
    
    return scores
# EVOLVE-BLOCK-END
