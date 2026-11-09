import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function for minimization.
    
    Instead of calculating a deterministic expected improvement, we sample
    from the Gaussian Process posterior for each candidate point and select
    the point with the lowest sampled value. This is implemented by returning
    the negative of the sampled value as the score, so that the highest score
    corresponds to the lowest sampled function value.
    """
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Draw samples from the standard normal distribution
    z = np.random.normal(0, 1, size=mu.shape)
    
    # Compute the sampled objective values
    # f_sampled ~ N(mu, sigma^2)
    sampled_f = mu + sigma_safe * z
    
    # The evaluator picks the candidate with the highest score.
    # We want to minimize the function, so we want the candidate with the
    # lowest sampled value to have the highest score.
    scores = -sampled_f
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
