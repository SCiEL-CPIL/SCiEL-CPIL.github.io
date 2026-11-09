import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Thompson Sampling
    # Sample from the posterior
    # We want to find the point with the lowest expected value.
    # So we sample y ~ N(mu, sigma) and pick the min y.
    # The acquisition function returns scores, higher is better.
    # So we return -y.
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Generate random noise
    # Using a fixed seed would make it deterministic, but BO is stochastic.
    # The evaluator calls this function. We should use random noise.
    # However, if the evaluator runs this in a loop, the randomness is part of the strategy.
    
    noise = np.random.randn(len(mu))
    samples = mu + sigma_safe * noise
    
    # Higher score means "evaluate next".
    # We want the sample with the lowest value (best objective).
    # So we negate the samples.
    scores = -samples
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
