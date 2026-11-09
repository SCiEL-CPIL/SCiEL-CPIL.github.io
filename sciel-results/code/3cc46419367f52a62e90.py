import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Computes the Expected Improvement (EI) acquisition function for minimization.
    
    Parameters:
    mu : array-like, shape (N,)
        Posterior mean at N candidate points.
    sigma : array-like, shape (N,)
        Posterior standard deviation at N candidate points.
    f_best : float
        Current best observed function value.
        
    Returns:
    scores : ndarray, shape (N,)
        EI scores for the candidate points. Higher scores indicate more promising points.
    """
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Expected Improvement for minimization:
    # EI(x) = E[max(f_best - f(x), 0)]
    # where f(x) ~ N(mu(x), sigma(x)^2)
    #
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Compute EI
    from scipy.stats import norm
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # EI is already "higher is better" since we want to maximize improvement
    # But EI can be zero or negative for points where mu > f_best (unlikely to improve)
    # We return EI directly as scores
    
    return ei
