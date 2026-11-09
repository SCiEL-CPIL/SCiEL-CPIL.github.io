"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track time for safety margin
_start_time = None

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    Args:
        mu: numpy array of shape (N,) with GP mean predictions.
        sigma: numpy array of shape (N,) with GP standard deviation predictions.
        f_best: float, the best observed value so far.
        
    Returns:
        numpy array of shape (N,) with scores. Higher scores indicate better candidates.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Safety guard: if we are running out of time, fall back to a simple deterministic score
    # to avoid timeouts. Although TS is fast, this is a defensive measure.
    elapsed = time.time() - _start_time
    if elapsed > 320:
        return mu - sigma

    # Thompson Sampling:
    # We want to maximize the sampled function value.
    # Draw a single random number z ~ N(0, 1)
    # The sampled function value at each point is f_sampled = mu + sigma * z
    # Since z is the same for all points, ranking the points by f_sampled is equivalent to
    # ranking them by (mu + sigma * z).
    # To ensure "higher score means evaluate next", we return the sampled values.
    
    z = np.random.normal(0, 1)
    
    # Compute scores
    scores = mu + sigma * z
    
    # Ensure finite scores
    # If sigma is very small, this is just mu. If mu/sigma are weird, check finiteness.
    # The problem states mu and sigma are from a GP, so they should be finite.
    # But just in case, replace non-finite with a very low value to deprioritize them.
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
