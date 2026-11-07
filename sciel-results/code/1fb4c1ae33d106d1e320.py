import time
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Probability of Improvement (PI) for minimization.
    
    PI is a probabilistic acquisition function that estimates the probability 
    that a point x will have a function value lower than the current best f_best.
    For a GP posterior f(x) ~ N(mu, sigma^2), 
    PI(x) = P(f(x) < f_best) = Phi((f_best - mu) / sigma).
    
    This approach is structurally different from UCB as it is non-linear and 
    bounded, often providing a better balance between exploration and 
    exploitation for multimodal functions by focusing on regions with high 
    uncertainty that are close to the current best performance.
    """
    start_time = time.time()
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-scores
    # z = (f_best - mu) / sigma
    # We want to maximize the probability of f(x) < f_best
    z = (f_best - mu) / sigma_safe
    
    # Calculate Probability of Improvement
    # norm.cdf(z) gives P(Z < z) for standard normal Z
    scores = norm.cdf(z)
    
    # Ensure all scores are finite
    if not np.all(np.isfinite(scores)):
        scores = np.where(np.isfinite(scores), scores, 0.0)
        
    # Time guard
    if time.time() - start_time > 290:
        return scores

    return scores
