# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is non-negative and handle potential NaNs/Infs in inputs
    sigma = np.maximum(sigma, 0.0)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Draw independent standard normal samples for each candidate
    # This approximates Thompson Sampling without the full covariance matrix
    z = np.random.standard_normal(size=mu.shape)
    
    # Draw samples from the GP posterior
    # sample = mu + sigma * z
    samples = mu + sigma * z
    
    # Center the samples relative to the current best value
    # This makes the acquisition function scale-invariant with respect to f_best
    # and focuses on potential improvement (regret)
    samples_relative = samples - f_best
    
    # Cost-aware score: Minimize (sample_relative / cost)
    # Since we want to return scores where HIGHER is better, we negate the ratio
    # If samples_relative is negative (better than f_best), score is positive.
    # If samples_relative is positive (worse), score is negative.
    
    score = -samples_relative / cost_safe
    
    # Ensure all scores are finite
    # Replace non-finite values with a very low score (or -inf) to avoid selection
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END
