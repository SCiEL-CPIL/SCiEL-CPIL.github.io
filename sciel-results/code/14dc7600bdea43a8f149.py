import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 300.0 # Safety margin from 330s

def acquisition(mu, sigma, f_best):
    # Check time
    current_time = time.time()
    elapsed = current_time - _START_TIME
    
    # If we are close to the limit, return a simple score to avoid timeout
    # The evaluator kills at 330s. We should finish well before.
    # If elapsed > 300s, we are in danger.
    # But this function is called many times.
    # If we are already over, we can't do much.
    # The instruction says "return the best result found with a safety margin".
    # This usually applies to the main loop, not the acquisition function.
    # However, if the acquisition function is slow, it contributes to the time.
    # Here it is fast.
    
    # Proceed with PI
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    pi = norm.cdf(z)
    
    return pi
