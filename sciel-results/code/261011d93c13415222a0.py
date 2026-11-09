"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 295  # Seconds - leave safety margin before 300s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Adjusted Expected Improvement (EI).
    
    This function calculates the Expected Improvement for each candidate point 
    and normalizes it by the evaluation cost. This directly optimizes the 
    expected reduction in the objective function per unit of cost, which is 
    ideal for the cost-aware setting described in the task.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP posterior mean in objective units
    sigma : array-like, shape (2000,)
        GP posterior std in objective units
    f_best : float
        Lowest observed objective value
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Deadline guard
    if time.time() - _start_time > _TIME_LIMIT:
        return np.zeros_like(np.asarray(mu, dtype=np.float64))
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is strictly positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    # Ensure cost is strictly positive
    cost = np.maximum(cost, 1e-9)
    
    # Calculate z-score: (f_best - mu) / sigma
    # For minimization, we want mu < f_best.
    # z > 0 indicates mu is lower than f_best (potential improvement).
    z = (f_best - mu) / sigma
    
    # Expected Improvement formula for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Where Phi is CDF and phi is PDF of the standard normal distribution.
    
    # To handle numerical stability for very large/small z:
    # If z is very large positive, Phi(z) ~ 1, phi(z) ~ 0. EI ~ f_best - mu.
    # If z is very large negative, Phi(z) ~ 0, phi(z) ~ 0. EI ~ 0.
    # NumPy's norm.cdf and norm.pdf are generally stable, but we can clip z 
    # to prevent any extreme floating point issues if necessary, though 
    # standard implementations handle this well.
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure EI is non-negative (numerical noise might cause tiny negatives)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: Expected Improvement per unit cost
    scores = ei / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
