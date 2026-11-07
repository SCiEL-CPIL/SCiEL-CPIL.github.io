# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_deadline = None
_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    """
    Compute a cost-aware acquisition score using Thompson Sampling.
    Draw a single sample from the posterior: y = mu + sigma * N(0,1)
    Score = -y / cost (higher is better, so minimize y/cost)
    """
    global _deadline, _start_time, _call_count
    
    if _start_time is None:
        _start_time = time.time()
        _deadline = _start_time + 325.0  # 5s safety margin before 330s limit

    # Increment call count
    _call_count += 1

    # Check deadline
    if time.time() > _deadline:
        # Emergency fallback: return simple EI
        eps = 1e-10
        sigma_safe = np.maximum(sigma, eps)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        return np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)

    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Thompson Sampling: draw one sample from the posterior
    # y = mu + sigma * z where z ~ N(0,1)
    z = np.random.standard_normal(len(mu))
    y = mu + sigma_safe * z
    
    # Score = -y / cost (we want to minimize y, and prefer low cost)
    # Higher score = better
    scores = -y / cost_safe
    
    # Handle numerical issues
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores

# EVOLVE-BLOCK-END
