"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None
_eval_count = 5  # Start counting after the initial 5 random points

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of hartmann 6d.
    Uses an adaptive Upper Confidence Bound (UCB) strategy for minimization.
    The exploration coefficient beta decays over time to shift from exploration to exploitation.
    
    Parameters:
    mu: array of shape (2000,) - posterior mean
    sigma: array of shape (2000,) - posterior standard deviation
    f_best: float - best observed value (minimum)
    
    Returns:
    scores: array of shape (2000,) - acquisition scores (higher is better)
    """
    global _start_time, _eval_count
    
    if _start_time is None:
        _start_time = time.time()
        
    # Increment the evaluation counter
    _eval_count += 1
    
    # Safety margin: If the total time exceeds 280s, we return a simplified score
    # to ensure we don't risk a timeout in the evaluator's loop.
    if time.time() - _start_time > 280:
        # Fall back to a pure exploitation strategy (minimize mu)
        # to finish quickly.
        scores = -mu
        return np.where(np.isfinite(scores), scores, -np.inf)

    # Ensure sigma is positive to avoid division by zero or invalid values
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Adaptive Beta Calculation
    # We have 25 decision steps (evaluations 6 through 30).
    # Let's map the step index to a beta value.
    # Step index: 0 to 24 corresponding to _eval_count 6 to 30.
    step_idx = _eval_count - 6  # 0-based index of the current decision step
    
    # Linear decay from high exploration to lower exploitation
    # Start beta at 3.5, end beta at 1.5
    # Total steps: 25
    if step_idx < 0:
        step_idx = 0
    if step_idx > 24:
        step_idx = 24
        
    beta = 3.5 - (2.0 * step_idx / 24.0)
    
    # UCB for minimization:
    # Score = beta * sigma - mu
    # Higher score means more promising.
    scores = beta * sigma_safe - mu
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
