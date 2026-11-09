"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline guard
_START_TIME = time.time()
_DEADLINE = 320.0  # 10s safety margin before 330s limit

def acquisition(mu, sigma, f_best):
    """
    Computes UCB acquisition scores.
    
    Args:
        mu: array of shape (N,), predicted means.
        sigma: array of shape (N,), predicted standard deviations.
        f_best: float, best function value observed so far.
        
    Returns:
        scores: array of shape (N,), higher values indicate more promising points.
    """
    # Check time limit
    if time.time() > _DEADLINE:
        # Fallback: return random scores or just -mu if we are running out of time
        # But typically this function is called many times per second, so this is a safety net.
        # If we are past deadline, we should probably just return something valid quickly.
        return -mu

    # UCB parameter: balance between exploitation and exploration.
    # For Beale function (narrow minimum, flat elsewhere), higher exploration is often needed.
    beta = 1.5
    
    # Calculate UCB score: -mu + beta * sigma
    # Higher score means we prefer this point.
    scores = -mu + beta * sigma
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
