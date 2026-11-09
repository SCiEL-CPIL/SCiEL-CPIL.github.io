"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 330.0  # 330s limit, use 325s for safety margin

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Probability of Improvement (POI) acquisition function.
    
    Args:
        mu: Array of GP mean predictions (shape 2000,)
        sigma: Array of GP standard deviations (shape 2000,)
        f_best: Scalar, lowest observed value
        cost: Array of evaluation costs (shape 2000,)
        
    Returns:
        Array of acquisition scores (shape 2000,), higher is better
    """
    # Check deadline (though this function is fast, it's good practice per directive)
    if time.time() > _DEADLINE - 5.0:
        # If we're close to the deadline, return a safe default
        # Ideally, the caller should handle this, but we ensure we don't hang.
        # Returning zeros is safe (won't pick any point if all are 0, or picks first if ties).
        # However, we must return finite scores.
        return np.zeros_like(mu)

    # Ensure sigma is non-negative and avoid division by zero
    # sigma is a standard deviation, so it should be >= 0.
    # Add a small epsilon to prevent division by zero.
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate z-score for Probability of Improvement
    # z = (f_best - mu) / sigma
    # If mu < f_best, z > 0, POI > 0.5
    # If mu > f_best, z < 0, POI < 0.5
    z = (f_best - mu) / sigma_safe
    
    # Calculate POI
    poi = norm.cdf(z)
    
    # Ensure poi is within [0, 1] just in case of numerical issues
    poi = np.clip(poi, 0.0, 1.0)
    
    # Cost-aware score
    # cost is 1 + mean(x), so it's > 1.
    # We want to maximize POI / cost
    # Ensure cost is positive to avoid division by zero or negative scores
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = poi / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
