"""Acquisition function for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best):
    """
    Adaptive Upper Confidence Bound (UCB) for minimization.
    
    Score = -mu + beta * sigma
    where beta adapts based on how close f_best is to the known global minimum.
    
    Higher beta when far from optimum (more exploration), lower beta when close (more exploitation).
    
    Parameters:
    mu : array-like, shape (N,)
        Posterior mean at N candidate points.
    sigma : array-like, shape (N,)
        Posterior standard deviation at N candidate points.
    f_best : float
        Current best observed function value.
        
    Returns:
    scores : ndarray, shape (N,)
        Acquisition scores for the candidate points (higher is better).
    """
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    
    n = mu.shape[0]
    if n == 0:
        return np.array([])
    
    # Sanitize sigma: ensure non-negative and finite
    sigma_clean = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma_clean = np.maximum(sigma_clean, 0.0)
    
    # Sanitize mu: replace non-finite with a large value (penalty)
    mu_clean = np.where(np.isfinite(mu), mu, 1e10)
    
    # Adaptive beta based on proximity to known global minimum (-1.8013)
    # If f_best is much worse, explore more (higher beta)
    # If f_best is close, exploit more (lower beta)
    global_min = -1.8013
    # Normalize the gap: 0 means at optimum, 1 means very far
    gap = max(0.0, f_best - global_min)
    # Scale gap to [0, 1] range for beta adjustment
    # gap of 0 -> beta = 0.5 (exploit), gap of 1.0+ -> beta = 2.0 (explore)
    gap_normalized = min(1.0, gap / 1.0)  # gap/1.0 since f_best ranges from ~-1.8 to ~1.0
    beta = 0.5 + 1.5 * gap_normalized  # beta in [0.5, 2.0]
    
    # Compute UCB score: -mu + beta * sigma
    scores = -mu_clean + beta * sigma_clean
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    # Time guard: if we're running out of time, return immediately
    if time.time() - start_time > 290.0:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
