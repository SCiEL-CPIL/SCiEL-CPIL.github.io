import numpy as np
import time

# Global state for deadline tracking
_start_time = None

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Lower Confidence Bound (LCB).
    
    Args:
        mu: (2000,) array of GP means.
        sigma: (2000,) array of GP standard deviations.
        f_best: float, the best observed value so far.
        
    Returns:
        (2000,) array of scores. Higher scores indicate more promising candidates.
    """
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
        
    # Deadline guard: 
    # The total wall clock limit is 330s.
    # We assume this function is called 90 times (3 seeds * 30 steps).
    # Average budget per call is ~3.6s.
    # Our computation is O(N) and very fast (<1ms).
    # We add a check just in case, though it's unlikely to trigger.
    elapsed = time.time() - _start_time
    if elapsed > 325.0: # Safety margin
        # Return a safe fallback, e.g., EI or just -mu
        # Using -mu (greedy) is a safe fallback
        return -mu

    # LCB Strategy
    # We want to minimize f.
    # LCB(x) = mu(x) - beta * sigma(x)
    # We want to find the x with the lowest LCB.
    # The acquisition function must return a score where HIGHER is better.
    # So Score(x) = -LCB(x) = -mu(x) + beta * sigma(x)
    
    # Beta controls exploration vs exploitation.
    # beta = 0 -> Greedy (-mu)
    # beta -> inf -> Pure Exploration (sigma)
    # For Hartmann 3D, a moderate beta is appropriate.
    beta = 2.5
    
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
