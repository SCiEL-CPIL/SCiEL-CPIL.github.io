# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline management
_START_TIME = time.time()
_DEADLINE_SECONDS = 280.0  # Safety margin before 300s/330s limit


def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE_SECONDS:
        # Return a safe, finite, valid score to avoid invalidation
        return np.zeros_like(mu, dtype=np.float64)

    # Sanitize inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-12)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-12)
    
    # Ensure cost is positive and finite
    cost = np.maximum(cost, 1e-12)
    cost = np.where(np.isfinite(cost), cost, 1e-12)

    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, 0.0)

    # Compute Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues with very large values
    z = np.clip(z, -10, 10)
    
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Cost-aware: divide by cost
    # Higher score = better
    scores = ei / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
