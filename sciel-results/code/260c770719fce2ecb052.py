"""Acquisition function for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 320.0  # safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for minimizing branin 2d.
    
    Parameters:
    -----------
    mu : ndarray of shape (2000,)
        GP mean predictions at candidate points (in objective units)
    sigma : ndarray of shape (2000,)
        GP standard deviation predictions at candidate points (in objective units)
    f_best : float
        Best observed objective value so far (minimum)
    
    Returns:
    --------
    scores : ndarray of shape (2000,)
        Acquisition scores; higher means more promising to evaluate next.
    """
    global _start_time, _DEADLINE
    
    # Check deadline
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Return a safe fallback: just use negative mean (exploitation)
        return -mu.copy()
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle non-finite values
    mask = np.isfinite(mu) & np.isfinite(sigma)
    if not np.all(mask):
        mu = np.where(mask, mu, f_best)
        sigma = np.where(mask, sigma, 1e-6)
    
    # Clip sigma to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute UCB with adaptive beta
    # Branin is multimodal, so we want to balance exploration and exploitation
    # Use a beta that adapts based on how many evaluations we've done
    # Since we don't know the step directly, we infer from f_best magnitude
    # or just use a fixed reasonable beta
    
    # For Branin, a good balance is around beta=1.0 to 2.0
    # Let's use a slightly more exploratory beta to escape local optima
    beta = 1.5
    
    # UCB: mu + beta * sigma
    # Since we're minimizing, we want to maximize the acquisition
    # Higher scores mean we should evaluate that point
    scores = mu + beta * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
