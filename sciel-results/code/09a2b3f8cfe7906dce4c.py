# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_DEADLINE = 330.0
_START_TIME = None

def _check_deadline():
    """Return True if we should stop and return current best."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return (time.time() - _START_TIME) > 325.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI).
    
    Parameters:
    - mu: array of shape (2000,), GP mean predictions
    - sigma: array of shape (2000,), GP standard deviation predictions
    - f_best: float, lowest observed value
    - cost: array of shape (2000,), evaluation costs
    
    Returns:
    - array of shape (2000,), finite scores (higher = evaluate next)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: replace any non-finite or invalid values
    # Ensure sigma is positive
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-12), sigma, 1e-9)
    # Ensure cost is positive
    cost = np.where(np.isfinite(cost) & (cost > 1e-12), cost, 1.0)
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Calculate Expected Improvement (EI)
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    # Avoid division by zero
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues in norm.cdf/pdf for extreme values
    # norm.cdf(-10) is ~0, norm.pdf(-10) is ~0
    # norm.cdf(10) is ~1, norm.pdf(10) is ~0
    # This is safe for standard normal distribution calculations
    z = np.clip(z, -10, 10)
    
    phi_z = norm.pdf(z)
    cdf_z = norm.cdf(z)
    
    # EI calculation
    # If mu < f_best, (f_best - mu) is positive, z is positive.
    # If mu > f_best, (f_best - mu) is negative, z is negative.
    ei = (f_best - mu) * cdf_z + sigma * phi_z
    
    # EI should be non-negative. Due to numerical precision, it might be slightly negative.
    ei = np.maximum(ei, 0.0)
    
    # Cost-adjusted score: Higher EI per unit cost is better
    score = ei / cost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
