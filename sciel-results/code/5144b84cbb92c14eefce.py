import numpy as np

def acquisition(mu, sigma, f_best):
    # Thompson Sampling
    # We simulate a sample from the posterior for each candidate
    # and select the one with the lowest value.
    # Since the evaluator selects the highest score, we return -sample.
    
    # Ensure sigma is positive to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Draw standard normal random numbers
    # Use a fixed seed? No, the task implies stochasticity is allowed/expected for TS.
    # However, for reproducibility in debugging, one might worry. 
    # But BO is stochastic anyway.
    z = np.random.randn(len(mu))
    
    # Sample from N(mu, sigma^2)
    y = mu + sigma_safe * z
    
    # We want to minimize y.
    # Return -y so that the minimum y has the maximum score.
    return -y
