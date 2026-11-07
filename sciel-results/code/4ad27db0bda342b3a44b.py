"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    # Safety guards
    if len(mu) != len(sigma):
        return np.zeros(len(mu))
    
    # Thompson Sampling:
    # Draw a single sample from the posterior distribution: theta = mu + sigma * Z
    # where Z is standard normal noise.
    # Score is -theta (since we want to minimize, lower theta is better, so higher score is -theta).
    
    # Ensure sigma is non-negative and safe for multiplication
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Generate standard normal noise for all 2000 points
    # Using a fixed seed based on f_best or time is not ideal for reproducibility in BO,
    # but Thompson sampling inherently requires randomness. 
    # The evaluator calls this function in a loop. The randomness helps escape local optima.
    # We use np.random.randn for the noise.
    z = np.random.randn(len(mu))
    
    # Sample from the posterior
    theta = mu + sigma_safe * z
    
    # Score: higher is better for selection.
    # We want to select the point with the lowest expected value (theta).
    # So we maximize -theta.
    scores = -theta
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
