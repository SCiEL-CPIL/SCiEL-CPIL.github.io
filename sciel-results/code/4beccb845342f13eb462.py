"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track elapsed time and enforce deadline
_start_time = time.time()
_DEADLINE = 300.0  # 300 seconds limit per evaluation


def acquisition(mu, sigma, f_best, cost):
    """
    Hybrid UCB acquisition function.
    
    Args:
        mu: array of shape (2000,), predicted mean
        sigma: array of shape (2000,), predicted standard deviation
        f_best: scalar, best observed value
        cost: array of shape (2000,), evaluation costs
    
    Returns:
        array of shape (2000,), acquisition scores (higher is better)
    """
    # Check deadline
    if time.time() - _start_time > _DEADLINE - 5.0:
        # Return a simple fallback if we're close to timeout
        return -mu
    
    # UCB: f_best - mu + beta * sigma
    # We use a fixed beta value. The knowledge suggests UCB-like hybrid improved score.
    # Let's try beta = 1.0 or slightly higher to encourage exploration.
    # In 2D Ackley, exploration is key to find the global minimum.
    beta = 1.0
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate UCB score
    # Higher score means more attractive
    # We want to minimize f, so lower mu is better.
    # UCB for minimization: -mu + beta * sigma
    # But we can also anchor it to f_best to make scores comparable.
    # Standard UCB for minimization: score = -mu + beta * sigma
    # Alternatively, use f_best - mu + beta * sigma to keep scale similar to EI.
    scores = f_best - mu + beta * sigma
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
