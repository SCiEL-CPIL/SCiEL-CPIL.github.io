import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    Samples from the posterior predictive distribution at each candidate point.
    Returns the negative of the sampled values so that the maximum score
    corresponds to the minimum sampled function value.
    """
    # Ensure sigma is positive to avoid issues with sqrt or scaling
    sigma = np.maximum(sigma, 1e-8)
    
    # Draw random numbers from the standard normal distribution
    # The shape matches the number of candidates
    z = np.random.randn(len(mu))
    
    # Sample from the Gaussian posterior N(mu, sigma^2)
    samples = mu + sigma * z
    
    # We want to minimize the function, so we return the negative of the samples.
    # The evaluator will pick the candidate with the highest score (lowest sample).
    return -samples
