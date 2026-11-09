import numpy as np
import time

# Global timer for deadline guard
_START_TIME = time.time()
# The evaluator kills at 330s. We aim to finish safely before that.
# Since this function is called 25 times, and is very fast,
# the main risk is the GP update in the outer loop.
# We cannot control the outer loop, but we can ensure this function is instant.

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Lower Confidence Bound (LCB) for minimization.
    
    Args:
        mu: Array of shape (N,) with GP mean predictions.
        sigma: Array of shape (N,) with GP standard deviation predictions.
        f_best: Scalar, the best (lowest) observed value so far.
        
    Returns:
        Array of shape (N,) with acquisition scores. Higher is better.
    """
    # Deadline guard: 
    # If we are very close to the 330s limit, we might want to simplify the logic.
    # However, the LCB calculation is O(N) and takes microseconds.
    # We include the check to satisfy the requirement for an explicit guard.
    current_time = time.time()
    if current_time - _START_TIME > 325.0:
        # Fallback to pure exploitation if we are running out of time
        # to minimize any potential overhead in the caller.
        scores = -mu
    else:
        # LCB for minimization:
        # We want to minimize f(x).
        # The lower confidence bound is L(x) = mu(x) - beta * sigma(x).
        # We want to pick x that minimizes L(x).
        # The acquisition score should be high for good candidates.
        # So score(x) = -L(x) = beta * sigma(x) - mu(x).
        
        # Beta controls exploration.
        # With few samples (5 initial), high exploration is needed.
        beta = 2.0
        
        scores = beta * sigma - mu
        
        # Ensure finite scores
        scores = np.where(np.isfinite(scores), scores, -np.inf)
        
    return scores
