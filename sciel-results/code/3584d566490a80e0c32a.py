import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of a 2D RBF mixture.
    Uses Probability of Improvement (PI) to aggressively exploit regions
    likely to improve upon the current best observed value f_best.
    """
    # Ensure sigma is non-negative and replace very small values with epsilon
    # to prevent division by zero when the GP is certain at a point.
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate the standardized improvement: (f_best - mu) / sigma
    # This represents how many standard deviations the current mean is below f_best.
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best) = Phi(z)
    # This value is in [0, 1]. Higher values mean higher probability of improvement.
    scores = norm.cdf(z)
    
    # Ensure scores are finite
    # norm.cdf should always return finite values for finite inputs,
    # but we guard against any numerical issues.
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
