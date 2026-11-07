"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    # Ensure sigma is positive to avoid division issues if used, 
    # though here we just add it to mu.
    # Standard deviation should be non-negative.
    sigma = np.maximum(sigma, 0.0)
    
    # Thompson Sampling:
    # Draw a sample from the posterior N(mu, sigma^2)
    # Since sigma is the standard deviation, we add sigma * N(0, 1)
    
    # Generate standard normal random numbers
    # Using a fixed seed would make it deterministic per call, but Thompson Sampling
    # typically uses fresh randomness. The task says "fresh uniform candidates" 
    # are drawn by the evaluator, implying the state changes. 
    # However, the acquisition function is stateless. 
    # Using np.random is standard for TS in this context.
    
    samples = np.random.normal(0.0, 1.0, size=mu.shape)
    sampled_values = mu + sigma * samples
    
    # We want to minimize the function.
    # The candidate with the lowest sampled value is the most promising.
    # We need to return scores such that the highest score is picked.
    # So, we negate the sampled values.
    scores = -sampled_values
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
