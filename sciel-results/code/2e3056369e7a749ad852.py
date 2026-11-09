import numpy as np
import time
from scipy.stats import norm

# Global timer for safety
_start_time = time.time()
_DEADLINE = 320.0  # 10s margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    
    Uses adaptive Upper Confidence Bound (UCB) for minimization.
    Score = -mu + beta * sigma
    
    The beta parameter is adapted based on f_best. 
    Early in the optimization (high f_best), beta is high to encourage exploration.
    Late in the optimization (low f_best), beta decreases to favor exploitation 
    and converge to the global minimum.
    """
    # Check time budget
    if time.time() - _start_time > _DEADLINE:
        return np.zeros_like(mu)

    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-10)
    
    # Adaptive beta:
    # Weierstrass min is 0. 
    # If f_best is ~0.5 (early), beta ~ 2.5 (similar to previous fixed value).
    # If f_best is ~0.0 (late), beta ~ 1.0 (more exploitation).
    # Clip f_best to [0, 1] to be safe, though it should be in this range.
    fb = np.clip(f_best, 0.0, 1.0)
    beta = 1.0 + 3.0 * fb
    
    # UCB score for minimization:
    # We want to find points with low mu (good) and high sigma (uncertain)
    # score = -mu + beta * sigma
    score = -mu + beta * sigma
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
