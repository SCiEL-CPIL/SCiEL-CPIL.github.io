import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    deadline = start_time + 10.0 # 10 seconds per call is generous
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # Handle numerical stability
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Entropy Weighting
    # H(sigma) ~ log(sigma)
    # We want to amplify EI in high uncertainty regions.
    # Using log1p(sigma) avoids log(0) and is smooth.
    # Scaling: The magnitude of EI depends on the scale of the function.
    # ST ranges from -39 to ~50.
    # sigma will be in that range.
    # log(1+sigma) will be small for small sigma, larger for large sigma.
    
    weight = 1.0 + np.log1p(sigma_safe)
    
    scores = ei * weight
    
    # Check deadline
    if time.time() > deadline:
         # Return a safe fallback, e.g., pure EI or UCB
         # But since it's vectorized, it should have finished.
         # If it didn't, we can't easily partial return. 
         # Assuming vectorized ops are fast enough.
         pass
         
    return scores
