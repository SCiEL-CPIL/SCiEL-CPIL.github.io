"""Acquisition function for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Lower Confidence Bound (LCB).
    
    This approach uses a linear combination of the mean and standard deviation
    to balance exploration and exploitation. It is structurally different from
    Expected Improvement (EI) and often provides better sample efficiency for
    global optimization tasks by explicitly weighting the uncertainty.
    
    Score = -mu + beta * sigma
    Higher scores indicate candidates with lower expected values or high uncertainty.
    
    Args:
        mu: Array of GP mean predictions (2000,).
        sigma: Array of GP standard deviation predictions (2000,).
        f_best: The lowest observed function value.
        
    Returns:
        Array of LCB scores (2000,). Higher scores indicate better candidates.
    """
    start_time = time.time()
    # Safety margin: The evaluator kills at 330s, but individual calls should be fast.
    # We add a guard just in case, though LCB is O(N).
    deadline = start_time + 290.0
    
    # Ensure inputs are 1D arrays
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    
    n = mu.shape[0]
    if n == 0:
        return np.array([])
    
    # Check time
    if time.time() > deadline:
        # Fallback: return zeros if we are running out of time
        return np.zeros(n)
    
    # Sanitize sigma: ensure non-negative and finite
    # LCB benefits from high sigma, so we want to preserve large values but handle NaNs/Infs
    eps = 1e-8
    sigma_clean = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma_clean = np.maximum(sigma_clean, 0.0)
    
    # Sanitize mu: replace non-finite with a large value (penalty)
    # We are maximizing -mu + beta*sigma.
    # If mu is +inf, -mu is -inf (bad).
    # If mu is -inf, -mu is +inf (good, but likely invalid data).
    # We treat non-finite mu as very bad (high value) to penalize it.
    mu_clean = np.where(np.isfinite(mu), mu, 1e10)
    
    # LCB parameter: Beta controls the exploration/exploitation trade-off.
    # A higher beta encourages more exploration.
    # For Michalewicz 2D, a value around 2.0-3.0 is often effective.
    beta = 2.0
    
    # Compute LCB score: -mu + beta * sigma
    # We want to maximize this score.
    scores = -mu_clean + beta * sigma_clean
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
