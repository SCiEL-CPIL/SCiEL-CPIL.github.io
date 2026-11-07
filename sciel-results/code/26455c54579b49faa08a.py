import numpy as np

def acquisition(mu, sigma, f_best):
    # Use Thompson Sampling
    # Sample one value from the posterior distribution for each candidate
    # mu: array of shape (2000,)
    # sigma: array of shape (2000,)
    
    # Ensure sigma is non-negative for safety, though it should be from a GP
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Draw samples
    samples = np.random.normal(loc=mu, scale=sigma_safe)
    
    # The objective is to minimize the function value.
    # The evaluator selects the candidate with the highest score.
    # Therefore, we return the negative of the sampled values.
    # This ensures that the lowest sampled value (best estimate) gets the highest score.
    return -samples
