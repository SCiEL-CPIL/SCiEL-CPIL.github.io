"""Rewritten program using Probability of Improvement (PI) for Bayesian Optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Start time for deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320  # 320s deadline, 10s margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian Optimization.
    Uses Probability of Improvement (PI) for minimization.
    PI(x) = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    This aggressively exploits regions where the GP predicts improvement.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: return zeros to avoid timeout
        return np.zeros_like(mu)
    
    # Ensure sigma is strictly positive to avoid division by zero
    eps = 1e-12
    safe_sigma = np.maximum(sigma, eps)
    
    # Calculate z-score for improvement
    z = (f_best - mu) / safe_sigma
    
    # Probability of Improvement: P(f(x) < f_best) = Phi(z)
    pi = norm.cdf(z)
    
    # Ensure finite scores
    pi = np.nan_to_num(pi, nan=0.0, posinf=1.0, neginf=0.0)
    
    return pi
# EVOLVE-BLOCK-END
