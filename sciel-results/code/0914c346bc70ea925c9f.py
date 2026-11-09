# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # 300s limit, 10s safety margin

def acquisition(mu, sigma, f_best):
    """
    Gaussian Process-Upper Confidence Bound (GP-UCB) acquisition function for minimization.
    
    UCB balances exploitation (low mu) and exploration (high sigma) with a tunable parameter beta.
    For minimization, we want to minimize mu and maximize sigma.
    Score = -mu + beta * sigma
    
    Parameters:
    mu: array of shape (N,)
    sigma: array of shape (N,)
    f_best: scalar, lowest observed value
    
    Returns:
    scores: array of shape (N,), higher values indicate better candidates
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: simple exploitation
        return -mu
    
    # Avoid division by zero or invalid values
    sigma_safe = np.where(sigma > 1e-10, sigma, 0.0)
    
    # UCB parameter. 
    # A fixed beta of 1.5 is a reasonable heuristic for ~30 iterations.
    # It encourages exploration of high-uncertainty regions while still favoring low means.
    beta = 1.5
    
    # Score: higher is better.
    # We want low mu (good) and high sigma (uncertain).
    # So we take -mu and add beta * sigma.
    scores = -mu + beta * sigma_safe
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
