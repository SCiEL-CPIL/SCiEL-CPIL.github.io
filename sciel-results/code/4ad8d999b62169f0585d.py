# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling (Kriging Believer) acquisition function.
    
    Instead of computing a deterministic score like EI or UCB, we sample a 
    single function realization from the posterior GP at all candidate points.
    The candidate with the lowest sampled value is selected.
    
    This approach naturally balances exploration and exploitation:
    - In areas of high uncertainty (high sigma), the sampled values vary widely,
      increasing the chance of finding a very low value (exploration).
    - In areas of low uncertainty (low sigma) and low mean (low mu), the sampled
      values cluster around the low mean, making them likely to be the minimum 
      (exploitation).
    """
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety check for time
    if time.time() - start_time > 325:
        return -mu

    # Kriging Believer: Sample one realization of the GP posterior at all points.
    # f(x) ~ N(mu(x), sigma(x)^2)
    # We generate standard normal noise and scale it by sigma.
    
    # Use a deterministic seed for reproducibility within the function call?
    # No, Thompson Sampling requires stochasticity to explore.
    # However, the evaluator calls this function multiple times.
    # We want independent samples for each step to ensure diverse exploration.
    
    # Generate standard normal random numbers
    # Shape matches mu (2000,)
    eps = np.random.randn(mu.shape[0])
    
    # Compute the sampled function values
    # The sampled value at each candidate point is:
    sampled_values = mu + sigma * eps
    
    # We want to MINIMIZE the function.
    # Thompson Sampling selects the point with the LOWEST sampled value.
    # The acquisition function must return scores where HIGHER means EVALUATE NEXT.
    # So we return the negative of the sampled values.
    scores = -sampled_values
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
