"""Acquisition function for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level state
_acq_state = {"step": 0}

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a hybrid strategy:
    - Early steps: high exploration (large beta in UCB)
    - Mid steps: balanced exploration/exploitation
    - Late steps: exploitation-focused (small beta)
    
    Uses a time-decaying beta with a floor to maintain some exploration.
    Score = -mu + beta(t) * sigma
    
    Args:
        mu: Array of GP mean predictions (2000,).
        sigma: Array of GP standard deviation predictions (2000,).
        f_best: The lowest observed function value.
        
    Returns:
        Array of scores (2000,). Higher scores indicate better candidates.
    """
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Ensure inputs are 1D arrays
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    
    n = mu.shape[0]
    if n == 0:
        return np.array([])
    
    # Time guard
    if time.time() > deadline:
        return np.zeros(n)
    
    # Sanitize sigma: ensure non-negative and finite
    sigma_clean = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma_clean = np.maximum(sigma_clean, 0.0)
    
    # Sanitize mu: replace non-finite with a large value (penalty)
    mu_clean = np.where(np.isfinite(mu), mu, 1e10)
    
    # Increment step counter
    _acq_state["step"] += 1
    t = _acq_state["step"]
    
    # Time-decaying beta: starts at 3.0, decays to 0.5 over 30 steps
    # This maintains a minimum exploration level to avoid premature convergence
    beta = 0.5 + 2.5 * max(0.0, 1.0 - t / 30.0)
    
    # UCB score: -mu + beta * sigma (higher is better for minimization)
    scores = -mu_clean + beta * sigma_clean
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    # Time guard before return
    if time.time() > deadline - 5.0:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
