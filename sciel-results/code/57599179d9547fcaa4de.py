import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    Parameters:
    mu : np.ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : np.ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        The best (lowest) observed objective value so far.
        
    Returns:
    scores : np.ndarray, shape (2000,)
        Scores for each candidate. Higher scores indicate more promising candidates.
    """
    # Draw samples from the predictive posterior N(mu, sigma^2)
    # We want to minimize the objective, so we prefer samples with low values.
    # The score should be high for low objective values.
    # Therefore, score = -sample.
    
    # np.random.normal loc=mu, scale=sigma
    # sigma is standard deviation.
    samples = np.random.normal(loc=mu, scale=sigma)
    
    # Higher score means "evaluate next".
    # Since we minimize, a sample of -10 is better than a sample of 10.
    # So -(-10) = 10 is a better score than -(10) = -10.
    scores = -samples
    
    return scores
