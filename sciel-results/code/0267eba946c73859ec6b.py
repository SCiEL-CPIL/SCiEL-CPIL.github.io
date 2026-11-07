import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Probability of Improvement (PI) acquisition function for minimization.
    
    Args:
    mu: array-like, shape (n,) - Mean of the Gaussian Process posterior.
    sigma: array-like, shape (n,) - Standard deviation of the Gaussian Process posterior.
    f_best: float - The best objective value observed so far.
    
    Returns:
    score: array-like, shape (n,) - Acquisition score, higher is better.
    """
    # Ensure sigma is not zero to avoid division errors
    # A small epsilon is added to prevent division by zero
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Calculate the Z-score for the Probability of Improvement
    # We want P(f(x) < f_best)
    # f(x) ~ N(mu, sigma^2)
    # P(f(x) < f_best) = P((f(x) - mu)/sigma < (f_best - mu)/sigma) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    
    # Compute the CDF of the standard normal distribution
    # This gives the probability that f(x) is less than f_best
    score = norm.cdf(z)
    
    # Ensure the output is finite
    # norm.cdf should return finite values for finite inputs, 
    # but we check just in case of extreme values causing numerical issues
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
