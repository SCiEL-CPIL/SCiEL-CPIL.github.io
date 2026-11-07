"""Cost-aware acquisition function for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Track time for safety margin
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320  # 320s out of 330s wall-clock limit

def _norm_cdf(x):
    """Vectorized normal CDF."""
    return 0.5 * (1.0 + np.vectorize(lambda z: norm.cdf(z))(x))

def _norm_pdf(x):
    """Vectorized normal PDF."""
    return norm.pdf(x)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: EI - lambda * cost.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best observed value so far
    cost : array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = better)
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a safe fallback: just use -mu (exploit best predicted minimum)
        return -mu.copy()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clamp sigma to avoid division by zero and ensure positivity
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware score: EI - lambda * cost
    # Using a small lambda to gently prefer cheaper evaluations
    lambda_cost = 0.1
    scores = ei - lambda_cost * cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
