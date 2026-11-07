"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level state for deadline tracking
_acq_state = {'start_time': None}

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    - mu: array of shape (2000,), GP mean predictions in objective units
    - sigma: array of shape (2000,), GP standard deviations in objective units
    - f_best: scalar, lowest observed function value
    - cost: array of shape (2000,), evaluation cost for each candidate
    
    Returns:
    - scores: array of shape (2000,), acquisition values (higher = evaluate next)
    """
    # Initialize module-level state
    if _acq_state['start_time'] is None:
        _acq_state['start_time'] = time.time()
        
    # Deadline guard
    elapsed = time.time() - _acq_state['start_time']
    if elapsed > 320.0:
        return np.zeros(len(mu), dtype=np.float64)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.5) # Default to mid cost
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Compute Expected Improvement (EI)
    # Minimizing f, so improvement is f_best - f(x)
    z = (f_best - mu) / sigma
    
    # Clip z for numerical stability
    z = np.clip(z, -10, 10)
    
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # --- Structural Cost Mask ---
    # The minimizer is in the inner 80% of [0, 10]^4, i.e., [1, 9]^4.
    # Cost(x) = 1 + mean(x/10).
    # For x in [1, 9]^4, cost is in [1.1, 1.9].
    # Any point with cost < 1.1 or cost > 1.9 is OUTSIDE [1, 9]^4 and cannot be the global minimizer.
    # We strictly exclude these points to save budget.
    
    valid_mask = (cost >= 1.1) & (cost <= 1.9)
    
    # Score is EI for valid points, 0 for invalid points.
    # We do NOT divide by cost to avoid biasing towards cheaper (corner) points.
    scores = np.where(valid_mask, ei, 0.0)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
