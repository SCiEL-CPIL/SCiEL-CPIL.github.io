import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) for minimization.
    
    Args:
        mu: numpy array of shape (N,) posterior mean.
        sigma: numpy array of shape (N,) posterior standard deviation.
        f_best: float, the best observed function value so far.
        
    Returns:
        numpy array of shape (N,) scores. Higher scores indicate better candidates.
    """
    # Avoid division by zero
    sigma_safe = np.where(sigma > 1e-10, sigma, 1.0)
    
    # For minimization, improvement = f_best - f(x)
    # EI = E[max(f_best - f(x), 0)]
    # where f(x) ~ N(mu, sigma^2)
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    z = (f_best - mu) / sigma_safe
    
    # Standard normal CDF and PDF
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # EI for minimization
    scores = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
