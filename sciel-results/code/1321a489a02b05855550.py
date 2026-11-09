import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Compute Thompson Sampling scores for candidate points.
    
    Parameters:
    mu : np.ndarray
        Mean of the GP posterior at candidate points, shape (N,)
    sigma : np.ndarray
        Standard deviation of the GP posterior at candidate points, shape (N,)
    f_best : float
        Current best (lowest) observed value.
        
    Returns:
    np.ndarray
        Thompson Sampling scores for each candidate, shape (N,)
    """
    # Ensure sigma is positive to avoid division by zero or invalid values
    sigma = np.maximum(sigma, 1e-9)
    
    # Draw a single random sample from the standard normal distribution
    # This single scalar is added to the mean scaled by the standard deviation
    # This creates a stochastic landscape for this specific iteration
    z = np.random.normal(0, 1)
    
    # The sampled function value at each candidate point
    # Higher values indicate better candidates (since we are minimizing, 
    # we want to find points where the sampled function is low)
    # Wait, Thompson sampling for minimization:
    # We sample f(x) ~ N(mu(x), sigma(x)^2).
    # We want to minimize f(x). So we should pick the x that minimizes the sampled f(x).
    # The evaluator picks the candidate with the HIGHEST score.
    # Therefore, we must return scores such that the best candidate (lowest sampled f) has the highest score.
    # So score = -sampled_f.
    
    sampled_f = mu + z * sigma
    
    # We want the evaluator to pick the point with the lowest sampled_f.
    # The evaluator picks the max score.
    # So we return -sampled_f.
    
    scores = -sampled_f
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
