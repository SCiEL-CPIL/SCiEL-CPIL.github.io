"""Rewritten acquisition for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Start time for deadline guard
_start_time = time.time()
# The evaluator kills after 330s. We allow some margin.
# However, this function is called 25 times (30 evals - 5 init).
# Each call should be very fast.
# We don't strictly need a deadline guard inside acquisition if it's vectorized,
# but we can add a simple check if we were doing heavy loops.
# For UCB/EI, it's just vector operations.

def acquisition(mu, sigma, f_best):
    """
    Compute the Upper Confidence Bound (UCB) acquisition function.
    
    UCB balances exploitation (low mean) and exploration (high std).
    Score = -mu + beta * sigma
    
    Parameters:
    mu : np.ndarray
        Posterior mean at candidate points, shape (N,).
    sigma : np.ndarray
        Posterior standard deviation at candidate points, shape (N,).
    f_best : float
        Best observed value so far (minimum).
    
    Returns:
    scores : np.ndarray
        UCB scores, shape (N,), higher is better.
    """
    # Beta parameter for UCB. 
    # A value of 1.0 is a standard choice. 
    # Styblinski-Tang 1D has multiple local minima. 
    # Exploration is crucial to find the global minimum at ~ +/-2.9.
    beta = 1.0
    
    # Ensure sigma is non-negative and handle numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # UCB for minimization:
    # We want to maximize the acquisition function.
    # Low mu is good (exploitation).
    # High sigma is good (exploration).
    scores = -mu + beta * sigma_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
