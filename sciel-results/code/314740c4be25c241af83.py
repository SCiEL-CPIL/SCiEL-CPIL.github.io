# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard variables
_START_TIME = time.time()
_TIME_LIMIT = 300.0  # 300s limit, keep margin

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Styblinski-Tang 2D.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed function value.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Check deadline
    if time.time() - _START_TIME > _TIME_LIMIT - 5.0:
         # Return a safe default if we are about to time out
         # This should rarely happen for vectorized ops
         return np.zeros_like(mu, dtype=np.float64)

    # Convert to numpy arrays for efficient computation
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-10)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-10)
    
    # Compute z-score for Expected Improvement
    # For minimization: z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Cost-aware scoring: divide by cost to get EI per unit cost
    # Use a stronger cost penalty (quadratic) to prefer low-cost points
    # Knowledge suggests cost**2 improved performance to 1.094
    scores = ei / (cost ** 2.0)
    
    # Handle any numerical issues
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
