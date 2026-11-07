import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 295  # safety margin before 300s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid of Expected Improvement (EI)
    and Probability of Improvement (PoI), normalized by cost.
    
    Args:
        mu: Mean of the GP posterior, shape (2000,)
        sigma: Standard deviation of the GP posterior, shape (2000,)
        f_best: Best observed function value
        cost: Cost of evaluating each candidate, shape (2000,)
        
    Returns:
        scores: Acquisition scores, shape (2000,)
    """
    # Timeout guard
    if time.time() > _DEADLINE:
        # Fallback: simple cost-aware exploitation
        mu = np.asarray(mu, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        cost_safe = np.maximum(cost, 1e-9)
        return -mu / cost_safe

    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best)
    Phi = norm.cdf(z)
    phi = norm.pdf(z)
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * Phi + sigma_safe * phi
    ei = np.maximum(ei, 0.0)
    
    # Hybrid score: combine EI and PoI
    # EI captures the magnitude of expected improvement
    # PoI captures the probability of any improvement
    # Weighting: 0.7 * EI + 0.3 * PoI (normalized)
    # Normalize PoI to be comparable to EI scale
    # EI can be larger than 1, PoI is in [0,1]
    # We use a simple weighted sum, then divide by cost
    
    # To make them comparable, we scale EI by a factor
    # Since EI is in objective units and PoI is probability,
    # we can use: score = (0.7 * ei + 0.3 * Phi) / cost
    
    # However, EI and PoI have different scales. 
    # A better approach: use EI as the primary driver, 
    # but add a small bonus for high PoI to encourage exploration
    # when EI is low but probability is decent.
    
    # Let's try: score = ei / cost + 0.1 * Phi / cost
    # This emphasizes EI but gives a boost to points with high PoI
    
    scores = (ei + 0.1 * Phi) / cost_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
