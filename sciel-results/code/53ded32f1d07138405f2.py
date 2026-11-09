"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global timer for the deadline guard
_start_time = time.time()
_DEADLINE = 325.0  # 325 seconds to be safe against the 330s kill switch

def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function.
    
    Parameters:
    mu: numpy array of shape (N,), posterior mean
    sigma: numpy array of shape (N,), posterior standard deviation
    f_best: float, best observed value so far
    
    Returns:
    numpy array of shape (N,), acquisition scores (higher is better)
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Return a deterministic fallback, e.g., pure exploitation
        return -mu
    
    # UCB formula for minimization:
    # We want to minimize f.
    # Exploitation: low mu is good.
    # Exploration: high sigma is good.
    # Score = beta * sigma - mu
    # Higher score means we should pick that candidate.
    
    # Tuning parameter beta. 
    # Branin is multimodal. A beta of ~2.5 is a robust choice for 2D BO.
    beta = 2.5
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    
    scores = beta * sigma - mu
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
