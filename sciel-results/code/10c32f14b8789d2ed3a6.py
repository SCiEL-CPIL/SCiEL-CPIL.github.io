import numpy as np
from scipy.stats import norm
import time

# Global timer for deadline guard
_MODULE_START_TIME = time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Probability of Improvement (PI).
    
    PI maximizes P(f(x) < f_best) = Phi((f_best - mu) / sigma).
    For minimization, this gives an aggressive exploitation signal that
    focuses on regions where the posterior probability of improvement is
    highest. Unlike EI, PI doesn't weight by the magnitude of improvement,
    making it more robust for small-budget optimization on multimodal
    landscapes where we want to quickly find any improvement.
    """
    # Deadline guard: if we are close to the 330s limit, return a safe, fast score
    if time.time() - _MODULE_START_TIME > 300:
        # Fallback: Return sigma (pure exploration) to ensure we don't crash
        return np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)

    # Handle sigma = 0 cases to avoid division by zero
    eps = 1e-10
    sigma_safe = np.where(sigma > eps, sigma, eps)
    
    # Z-score: (f_best - mu) / sigma
    # For minimization, we want P(f(x) < f_best)
    # If mu < f_best, z is positive, and Phi(z) is high
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: Phi(z)
    # This is the posterior probability that f(x) < f_best
    pi_score = norm.cdf(z)
    
    # Ensure finite outputs
    score = np.nan_to_num(pi_score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
