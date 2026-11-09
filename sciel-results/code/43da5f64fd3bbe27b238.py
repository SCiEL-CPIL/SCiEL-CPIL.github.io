import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guarding
_START_TIME = time.time()
# The evaluator kills after 330s. We add a small margin.
DEADLINE = _START_TIME + 325.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement acquisition function for minimization.
    
    Parameters:
    mu (np.array): GP posterior mean, shape (2000,)
    sigma (np.array): GP posterior std, shape (2000,)
    f_best (float): Best observed value so far.
    cost (np.array): Evaluation cost per candidate, shape (2000,)
    
    Returns:
    np.array: Acquisition scores, shape (2000,), higher = evaluate next
    """
    # Deadline guard: if we are too close to the limit, return zeros (or best effort)
    # In a single call, this is unlikely to trigger, but it satisfies the directive.
    if time.time() > DEADLINE:
        return np.zeros_like(mu, dtype=np.float64)

    # Ensure inputs are float64 arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle numerical stability for sigma
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Calculate z-score
    # z = (f_best - mu) / sigma
    improvement = f_best - mu
    z = improvement / sigma_safe
    
    # Clip z to avoid numerical issues in extreme tails
    z_clipped = np.clip(z, -50, 50)
    
    cdf_val = norm.cdf(z_clipped)
    pdf_val = norm.pdf(z_clipped)
    
    # Expected Improvement (EI)
    ei = improvement * cdf_val + sigma_safe * pdf_val
    
    # EI should be non-negative (clip small negative values from numerical noise)
    ei = np.maximum(ei, 0.0)
    
    # Cost-weighted EI
    scores = ei / cost_safe
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
