import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Computes the Expected Improvement (EI) acquisition function.
    
    EI encourages exploration of regions where the GP posterior mean is
    close to the current best and the uncertainty is high, but it primarily
    focuses on points that are likely to improve upon the current best.
    
    Parameters:
    -----------
    mu : numpy array
        Posterior mean of the GP at candidate points.
    sigma : numpy array
        Posterior standard deviation of the GP at candidate points.
    f_best : float
        The best (lowest) function value observed so far.
        
    Returns:
    --------
    scores : numpy array
        EI scores for each candidate point. Higher is better.
    """
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate the z-score
    # z = (f_best - mu) / sigma
    # Since we are minimizing, improvement is f_best - f(x)
    z = (f_best - mu) / sigma_safe
    
    # Compute Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Phi is the CDF of the standard normal distribution
    # phi is the PDF of the standard normal distribution
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure all scores are finite
    # EI should be non-negative, but numerical errors might cause tiny negatives
    # We clip to 0 to ensure we don't prefer points with negative EI (which is impossible)
    ei = np.maximum(ei, 0.0)
    
    # Replace any non-finite values (NaN, Inf) with 0
    scores = np.where(np.isfinite(ei), ei, 0.0)
    
    return scores
