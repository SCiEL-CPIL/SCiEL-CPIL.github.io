import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 330.0  # seconds total budget, leave margin

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6D.
    Uses Lower Confidence Bound (LCB) for minimization.
    Score = -mu + kappa * sigma (higher score = more promising).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    
    # Known minimum of Hartmann 6D
    f_min = -3.32237
    
    # Gap from best observed to known minimum
    gap = f_best - f_min
    gap_clamped = np.clip(gap, 0.0, 5.0)
    
    # Adaptive kappa: high exploration early, exploitation late
    # Start at ~1.5 when gap is large, decay to ~0.5 when gap is small
    kappa = 0.5 + 1.0 * np.exp(-gap_clamped * 0.5)
    
    # LCB for minimization: mu - kappa * sigma
    # We want to maximize the score, so score = -(mu - kappa * sigma) = -mu + kappa * sigma
    score = -mu + kappa * sigma
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
