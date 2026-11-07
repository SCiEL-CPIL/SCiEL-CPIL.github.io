import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Computes Thompson Sampling scores for Bayesian optimization.
    
    Parameters:
    mu (np.ndarray): Mean of the GP posterior at candidate points, shape (2000,).
    sigma (np.ndarray): Standard deviation of the GP posterior at candidate points, shape (2000,).
    f_best (float): The lowest observed value so far (not directly used in standard TS, but kept for signature).
    
    Returns:
    np.ndarray: Scores for each candidate, shape (2000,). Higher scores indicate more promising candidates.
    """
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-8)
    
    # Draw samples from the posterior
    # z ~ N(0, 1)
    z = np.random.randn(mu.shape)
    
    # Sampled function values
    # f_sample(x) = mu(x) + sigma(x) * z(x)
    f_samples = mu + sigma * z
    
    # We want to minimize f(x), so we maximize -f(x)
    # Return negative samples so that the "highest score" corresponds to the "lowest sampled value"
    scores = -f_samples
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
