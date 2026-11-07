import numpy as np
import time

# Global state for time budget management
_acquisition_start_time = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using adaptive UCB with time budget guard.
    
    Args:
    mu: array of mean predictions (shape 2000,)
    sigma: array of standard deviation predictions (shape 2000,)
    f_best: scalar, best observed value
    cost: array of evaluation costs (shape 2000,)
    
    Returns:
    scores: array of acquisition scores (shape 2000,), higher is better
    """
    global _acquisition_start_time
    
    # Initialize start time if not set
    if _acquisition_start_time is None:
        _acquisition_start_time = time.time()
    
    elapsed = time.time() - _acquisition_start_time
    
    # Safety margin: return simple heuristic if we're getting close to timeout
    # 330s total budget, keep 30s margin
    if elapsed > 300:
        # Return a simple cost-aware exploitation score
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        scores = (f_best - mu) / cost_safe
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        return scores
    
    # Adaptive beta based on elapsed time:
    # Start with higher exploration, decrease as we run out of time
    # Total budget ~330s, so scale beta from 3.0 to 1.5 over the duration
    progress = min(elapsed / 330.0, 1.0)
    beta = 3.0 - 1.5 * progress
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Calculate UCB score
    scores = (f_best - mu + beta * sigma) / cost
    
    # Sanitize output to ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
