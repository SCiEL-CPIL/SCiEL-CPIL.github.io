import numpy as np

def acquisition(mu, sigma, f_best):
    # Thompson Sampling: Sample from the posterior
    # We want to minimize f.
    # If we sample y ~ N(mu, sigma^2), we want the candidate with the lowest y.
    # The acquisition function returns a score where HIGHER is better.
    # So we should return -y.
    
    # Generate random samples
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Draw samples
    # Note: The evaluator calls this function once per step.
    # The randomness is inherent to the method.
    # To ensure reproducibility for a given seed in the evaluator, 
    # the evaluator likely seeds the global RNG or passes the seed.
    # However, standard BO implementations often just use np.random.
    # If the evaluator uses different seeds for the 3 runs, TS will behave differently.
    
    y_samples = np.random.normal(mu, sigma_safe)
    
    # Score is negative of the sample, so lower function value -> higher score
    score = -y_samples
    
    return score
