import numpy as np
import time
from scipy.stats import norm

# Global timer for safety
_start_time = time.time()
_DEADLINE = 320.0  # 10s margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    
    Uses Adaptive Upper Confidence Bound (UCB) for minimization.
    Score = -mu + beta * sigma
    
    The beta parameter is adapted based on the current best value (f_best).
    In the early stages (high f_best), beta is large to encourage exploration.
    In the later stages (low f_best), beta is small to encourage exploitation.
    """
    # Check time budget
    if time.time() - _start_time > _DEADLINE:
        return np.zeros_like(mu)

    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-10)
    
    # Estimate progress based on f_best.
    # Assume initial f_best is around 0.5 and target is 0.
    # If f_best is not decreasing as expected, this still provides a reasonable heuristic.
    f_ref = 0.5
    f_best_clamped = np.clip(f_best, 0.0, f_ref)
    
    # Progress: 0.0 when f_best is high (start), 1.0 when f_best is low (end)
    progress = 1.0 - (f_best_clamped / f_ref)
    progress = np.clip(progress, 0.0, 1.0)
    
    # Adaptive beta
    # Start with high exploration, end with low exploration
    beta_start = 3.5
    beta_end = 0.2
    
    # Linear interpolation
    beta = beta_start + (beta_end - beta_start) * progress
    
    # UCB Score for minimization: -mu + beta * sigma
    score = -mu + beta * sigma
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
