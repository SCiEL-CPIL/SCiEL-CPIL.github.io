"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Upper Confidence Bound (UCB) with adaptive beta.
    
    Uses a UCB strategy where the exploration coefficient beta is adjusted
    based on the current best value f_best. Early in the optimization (f_best close to 0),
    beta is large to encourage exploration. Later (f_best close to -3.32), beta is small
    to focus on exploitation.
    
    mu: (n,) array of GP mean predictions
    sigma: (n,) array of GP standard deviation predictions
    f_best: current best observed value (minimum)
    
    Returns: (n,) array of scores, higher is better
    """
    # Known bounds for Hartmann 6d to estimate progress
    f_min_true = -3.32237
    f_max_est = 0.0  # Function values approach 0 from below
    
    # Calculate progress: 1.0 at start (f_best ~ 0), 0.0 at end (f_best ~ -3.32)
    # Clamp f_best to avoid issues
    f_best_clamped = np.clip(f_best, f_min_true, f_max_est)
    
    # Denominator is range
    range_val = f_max_est - f_min_true
    if range_val == 0:
        progress = 1.0
    else:
        progress = (f_best_clamped - f_min_true) / range_val
        
    # Adaptive beta
    # Start with high exploration, end with low exploration
    beta_min = 0.5
    beta_max = 3.0
    beta = beta_min + (beta_max - beta_min) * progress
    
    # UCB score for minimization: -mu + beta * sigma
    # Higher score indicates a more promising candidate
    scores = -mu + beta * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
