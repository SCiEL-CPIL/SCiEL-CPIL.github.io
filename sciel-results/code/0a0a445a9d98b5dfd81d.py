import numpy as np
from scipy.stats import norm
import time

# Global time tracking for safety
_start_time = time.time()
_time_limit = 300  # Seconds

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Probability of Improvement (POI) acquisition function.
    
    POI is defined as Phi((f_best - mu) / sigma) / cost, where Phi is the
    standard normal CDF. This provides a bounded, monotonic exploitation
    signal that naturally suppresses exploration in high-uncertainty regions,
    focusing the tight budget on confirming the global minimum.
    
    Args:
        mu: (2000,) array of GP mean predictions.
        sigma: (2000,) array of GP standard deviation predictions.
        f_best: Scalar, the best objective value observed so far.
        cost: (2000,) array of evaluation costs (1 + mean(x)).
        
    Returns:
        (2000,) array of acquisition scores (higher = evaluate next).
    """
    # Time guard: if we're getting close to the limit, return a safe default
    elapsed = time.time() - _start_time
    if elapsed > _time_limit - 5:  # 5 second safety margin
        return np.zeros_like(mu)
    
    # Ensure sigma is positive to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # --- Cost-Aware POI ---
    # Standardized improvement: (f_best - mu) / sigma
    # This is the z-score for the probability that f(x) < f_best
    z = (f_best - mu) / sigma
    
    # POI = Phi(z), bounded in (0, 1)
    poi = norm.cdf(z)
    
    # Cost-aware: divide by cost to prioritize cheaper evaluations
    # Cost is in [1, 2], so this effectively gives 2x weight to cheap points
    score = poi / cost
    
    # Ensure output is finite and real
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
