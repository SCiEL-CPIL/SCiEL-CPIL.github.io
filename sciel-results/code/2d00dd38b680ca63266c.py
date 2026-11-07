import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    
    Thompson Sampling works by drawing a sample from the posterior distribution
    of the function values at the candidate points and selecting the point that
    minimizes this sample (for minimization tasks).
    
    Since the evaluator selects the candidate with the highest score, we return
    the negative of the sampled values.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is positive and finite to avoid issues with normal distribution
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Handle non-finite mu
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Draw a sample from the posterior: y ~ N(mu, sigma)
    # We use a fixed seed per call? No, Thompson Sampling relies on randomness.
    # The evaluator calls this function repeatedly. The randomness comes from np.random.
    # This is standard for Thompson Sampling in BO loops.
    samples = mu + sigma * np.random.normal(size=mu.shape)
    
    # We want to minimize the objective, so we want the smallest sample.
    # The evaluator picks the max score. So we return -sample.
    score = -samples
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
