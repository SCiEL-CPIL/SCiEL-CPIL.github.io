import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() > _DEADLINE:
        return 1.0 / (1.0 + cost)
    
    # Clip sigma for numerical stability
    sigma = np.clip(sigma, 1e-9, None)
    
    # Calculate Expected Improvement (EI)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scoring: EI divided by cost
    # This balances improvement potential against evaluation expense.
    # Unlike exponential gates, this provides a smooth, linear penalty
    # proportional to the cost, which is better suited for cases where
    # the optimal cost is not strictly low (e.g., uniform minimizer distribution).
    scores = ei / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
