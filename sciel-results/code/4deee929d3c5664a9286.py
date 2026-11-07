import time
import numpy as np
from scipy.stats import norm

_start_time = None

def acquisition(mu, sigma, f_best):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Check deadline. The total budget is 330s.
    # We don't know exactly how much time is left, but we can check if we are dangerously close.
    # However, since this function is fast, it's unlikely to be the cause of a timeout.
    # The timeout would be caused by the GP fitting in the evaluator.
    # If I add a guard that returns a default (e.g., random or based on mu only) when time is up,
    # it might degrade performance if triggered.
    # Given the speed of norm.cdf, I will skip the guard to avoid complexity, 
    # UNLESS the directive strictly requires it.
    # "Keep an EXPLICIT deadline guard"
    
    elapsed = time.time() - _start_time
    # If we have used > 320s, something is wrong or the GP is slow.
    # But we can't control the GP.
    # I will implement the guard as requested, but it will almost never trigger.
    if elapsed > 325:
        # Return a fallback score. E.g., based on mean (exploitative) or random.
        # To be safe, return scores that prioritize low mean.
        return -mu 

    z = (f_best - mu) / (sigma + 1e-9)
    return norm.cdf(z)
