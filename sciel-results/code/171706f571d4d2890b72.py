"""Cost-aware acquisition for griewank-2d BO using Expected Improvement (EI)
with sub-linear cost scaling. EI captures the magnitude of potential improvement,
which is more effective than PoI for minimizing functions like Griewank where
finding the global minimum (0) is the primary goal."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_START_TIME = time.time()
_TIME_LIMIT = 300.0  # seconds, with safety margin

def acquisition(mu, sigma, f_best, cost):
    """Cost-aware Expected Improvement acquisition function.
    
    Returns 2000 finite scores (higher = evaluate next).
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for improvement: z = (f_best - mu) / sigma
    # Since we are minimizing, improvement is f_best - mu
    z = (f_best - mu) / sigma
    
    # Expected Improvement (EI) for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Cost-aware scaling:
    # We divide by cost^alpha. 
    # alpha=1 is strict efficiency (EI per cost).
    # alpha=0.5 is softer, allowing exploration of high-cost regions if EI is high.
    # Given the minimum can be shifted to any location, a softer penalty 
    # prevents the search from being trapped in low-cost (origin) regions 
    # when the true minimum is elsewhere.
    alpha = 0.5
    cost_scaled = np.power(np.maximum(cost, 0.1), alpha)
    
    scores = ei / cost_scaled
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Safety check: if deadline is approaching, return quickly
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        # Return simple EI as fallback
        scores = np.where(np.isfinite(ei), ei, 0.0)
        
    return scores
# EVOLVE-BLOCK-END
