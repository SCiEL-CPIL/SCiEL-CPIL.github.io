# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: 330s wall-clock limit, use 325s as safety margin
_DEADLINE = 325.0
_START_TIME = time.time()

def _check_deadline():
    """Return True if we should abort and return current best."""
    return (time.time() - _START_TIME) > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function.
    
    Strategy:
    - Compute standard Expected Improvement: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    - Normalize by cost: Score = EI / cost
    - This maximizes the expected improvement per unit cost.
    
    This approach is more robust than LCB because it accounts for the 
    probability of improvement, not just a lower bound. Dividing by cost 
    ensures that expensive evaluations are only chosen if they offer 
    proportionally higher expected gains.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values and safe ranges
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Clip sigma to avoid division by zero or numerical instability
    # If sigma is extremely small, EI should be max(f_best - mu, 0)
    sigma = np.clip(sigma, 1e-6, 1e6)
    
    # Clip cost to avoid division by zero (cost is >= 1 theoretically, but be safe)
    cost = np.clip(cost, 1e-6, 100.0)
    
    # Calculate Expected Improvement
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    # Handle cases where sigma is very small to avoid overflow in z
    # If sigma < 1e-5, treat as deterministic
    small_sigma_mask = sigma < 1e-5
    
    # Initialize EI
    ei = np.zeros_like(mu)
    
    # For small sigma, EI is approx max(f_best - mu, 0)
    ei[small_sigma_mask] = np.maximum(f_best - mu[small_sigma_mask], 0.0)
    
    # For normal sigma, compute standard EI
    normal_sigma_mask = ~small_sigma_mask
    if np.any(normal_sigma_mask):
        mu_n = mu[normal_sigma_mask]
        sigma_n = sigma[normal_sigma_mask]
        
        z = (f_best - mu_n) / sigma_n
        # Clip z to prevent extreme values in norm.cdf/pdf (though norm handles it well)
        z = np.clip(z, -10, 10)
        
        phi_z = norm.pdf(z)
        Phi_z = norm.cdf(z)
        
        ei_n = (f_best - mu_n) * Phi_z + sigma_n * phi_z
        ei[normal_sigma_mask] = ei_n
    
    # Ensure EI is non-negative (numerical noise might make it slightly negative)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scoring: Maximize EI per unit cost
    # Higher score = better
    score = ei / cost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
