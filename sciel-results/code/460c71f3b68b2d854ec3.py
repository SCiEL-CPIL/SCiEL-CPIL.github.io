import time
import numpy as np

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_TIME_LIMIT = 290.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d.
    Returns scores for 2000 candidates. Higher score is better.
    """
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback to pure exploitation if time is running out
        return -mu

    # f_best is the minimum observed value.
    # For sphere on [-5, 5], max value is 25.
    # We use f_best to adapt the exploration-exploitation trade-off.
    # Early on (high f_best), we want more exploration (higher beta).
    # Late on (low f_best), we want more exploitation (lower beta).
    
    # Clip f_best to valid range [0, 25]
    f_best_val = float(f_best)
    if not np.isfinite(f_best_val):
        f_best_val = 25.0
        
    f_best_clamped = np.clip(f_best_val, 0.0, 25.0)
    
    # Adaptive Beta
    # Beta ranges from 2.0 (at f_best=0) to 4.0 (at f_best=25)
    # This encourages strong exploration when the best value is still poor.
    beta = 2.0 + 2.0 * (f_best_clamped / 25.0)
    
    # UCB for minimization: Minimize mu - beta * sigma
    # Equivalent to Maximize -mu + beta * sigma
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
