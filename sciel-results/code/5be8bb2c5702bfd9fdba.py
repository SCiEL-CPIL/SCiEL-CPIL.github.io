import numpy as np
import time

# Global start time for the entire evaluation process
# Note: This assumes the module is loaded once per evaluation run.
_start_time = time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Adaptive Upper Confidence Bound (UCB).
    
    Args:
        mu (np.ndarray): Posterior mean of shape (2000,).
        sigma (np.ndarray): Posterior standard deviation of shape (2000,).
        f_best (float): The lowest observed value so far.
        
    Returns:
        np.ndarray: Scores of shape (2000,), higher is better.
    """
    # Deadline guard: If we are close to the 330s limit, fall back to exploitation
    # to ensure we return a valid result quickly.
    if time.time() - _start_time > 310:
        # Return scores that favor the lowest mean (exploitation)
        # Adding a small noise to avoid ties? No, just -mu.
        return -mu

    # Adaptive UCB
    # Beta controls the balance between exploitation (-mu) and exploration (sigma).
    # As f_best decreases (approaches the minimum 0), we reduce exploration to converge.
    # We keep a baseline exploration level to avoid getting stuck in local minima.
    # f_best is expected to be in [0, ~1] for Weierstrass.
    beta = 1.5 + 1.0 * f_best
    
    # Calculate UCB scores
    # We are minimizing, so we want low mu.
    # Score = -mu + beta * sigma
    scores = -mu + beta * sigma
    
    # Handle any potential numerical issues (NaNs, Infs)
    # If sigma is 0 and mu is inf, etc.
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
