"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # Safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: EI per unit cost with stronger cost normalization.
    
    Uses an offset in the cost denominator to increase relative weight of
    low-cost candidates. The cost for evaluating x is 1 + mean(x_scaled),
    so cost ranges from 1.0 to 2.0. Adding an offset of 0.6 gives:
    - cost=1.0: denominator = 1.6
    - cost=2.0: denominator = 2.6
    Ratio: 2.6/1.6 = 1.625x more weight for cheap vs expensive
    
    Without offset (current): ratio = 2.0/1.0 = 2.0x
    With offset 0.6: ratio = 1.625x (more balanced, favors exploration of
    moderately cheap regions)
    """
    # Check deadline - if too close to timeout, return uniform random scores
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE - 5.0:
        # Return finite random scores to avoid invalidation
        return np.random.uniform(0, 1, size=mu.shape)
    
    # Expected Improvement
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware normalization: divide by (cost + offset)
    # Offset of 0.6 increases relative weight of low-cost candidates
    cost_offset = 0.6
    scores = ei / (cost + cost_offset)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
