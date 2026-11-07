"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI)
    with a linear cost penalty.
    
    This approach prioritizes points that offer high expected improvement
    relative to their evaluation cost. The linear penalty (1/cost) is less
    aggressive than the quadratic penalty used in previous versions, allowing
    the search to explore areas with slightly higher cost but potentially
    better function values.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean predictions from the GP in objective units.
    sigma : array-like, shape (2000,)
        Standard deviation predictions from the GP in objective units.
    f_best : float
        Best observed objective value so far (minimum).
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    score : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost_arr = np.asarray(cost, dtype=np.float64)
    
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost_arr, 1e-8)
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Standard normal CDF and PDF
    # Using scipy.stats for accuracy
    Phi_z = norm.cdf(z)
    phi_z = norm.pdf(z)
    
    # Expected improvement
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # EI is non-negative by definition, but clip to avoid negative values due to floating point errors
    ei = np.maximum(ei, 0.0)
    
    # Apply Linear Cost Penalty
    # Dividing by cost directly.
    # This balances the magnitude of improvement against the resource cost.
    score = ei / cost_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
