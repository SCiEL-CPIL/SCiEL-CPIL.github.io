import numpy as np

def acquisition(mu, sigma, f_best):
    # Thompson Sampling
    # Sample from the posterior for each candidate
    # We want to minimize, so we pick the candidate with the lowest sampled value.
    # The interface returns scores where higher is better.
    # So we return -sampled_value.
    
    # Ensure sigma is positive and not zero to avoid issues, though GP sigma should be > 0
    # Add a small epsilon to sigma to prevent division by zero if it were used in other ways, 
    # but for normal distribution, sigma=0 is just a point mass.
    
    # Clip sigma to a minimum value to avoid numerical issues if sigma is extremely tiny
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Generate samples
    # shape of mu is (2000,)
    samples = np.random.normal(mu, sigma_safe)
    
    # We want to select the point with the lowest value.
    # The evaluator picks the point with the HIGHEST score.
    # So score = -sample
    scores = -samples
    
    return scores
