import numpy as np
from scipy.stats import norm
import time

# Global variable to track start time for deadline guard
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
        
    # Deadline guard: If we are close to the 300s limit, return a safe result
    # The limit is 300s per evaluation, but the total script might have a limit.
    # The prompt says "Each evaluation must finish within 300 s".
    # The BO loop runs multiple evaluations.
    # If the *entire* run is killed after 330s, we should be careful.
    # However, acquisition is fast. The bottleneck is the GP updates in the loop,
    # which we don't control.
    # We will just ensure this function is fast.
    
    # Convert to numpy arrays for safety
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is not zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI) for minimization
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: If mu > f_best, z is negative, Phi(z) is small, EI is small.
    # If mu < f_best, z is positive, Phi(z) is large, EI is large.
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a small exploration term to avoid premature convergence
    # UCB-like exploration: we prefer higher sigma if mu is not clearly worse
    # Let's use a small weight for sigma
    exploration = 0.1 * sigma_safe
    
    # Total score = (EI + exploration) / cost
    scores = (ei + exploration) / cost
    
    # Handle any non-finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
