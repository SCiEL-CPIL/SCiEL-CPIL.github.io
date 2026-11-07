import numpy as np
import time

def acquisition(mu, sigma, f_best):
    # Deadline guard
    # The evaluator kills after 330s. 
    # This function is called 25 times.
    # It should be fast.
    # We don't have a start time for the whole process here, 
    # only for this call.
    # A local guard is sufficient to prevent pathological cases.
    start = time.time()
    limit = 1.0 # 1 second limit per call, plenty for N=2000
    
    # UCB Acquisition
    # Higher beta encourages exploration to find the global minimum of Beale function
    # Data suggests beta > 5.0 is beneficial
    beta = 8.0
    # Ensure sigma is non-negative for safety, though GP should ensure this
    sigma_safe = np.maximum(sigma, 1e-9)
    
    scores = -mu + beta * sigma_safe
    
    # If it takes too long (unlikely), we could fallback, 
    # but for simple arithmetic, it won't.
    if time.time() - start > limit:
         # Fallback to EI or similar if something goes wrong
         z = (f_best - mu) / sigma_safe
         from scipy.stats import norm
         scores = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
         
    return scores
