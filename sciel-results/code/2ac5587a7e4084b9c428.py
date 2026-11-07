import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Thompson Sampling
    # Sample from the posterior
    # Note: sigma is standard deviation
    # We sample y ~ N(mu, sigma)
    # To ensure reproducibility in the context of the loop, 
    # the evaluator likely calls this once per iteration.
    # Randomness is allowed.
    
    # Handle zero sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Sample
    y = mu + sigma_safe * np.random.randn(len(mu))
    
    # Calculate improvement
    improvement = f_best - y
    
    # Cost-aware score
    # Higher is better
    # If cost is high, score is lower.
    score = improvement / cost
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
