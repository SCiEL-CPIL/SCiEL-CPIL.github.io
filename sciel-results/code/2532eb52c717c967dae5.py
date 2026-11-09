import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) for minimization.
    
    Args:
        mu: Array of posterior means at candidate points (shape (2000,)).
        sigma: Array of posterior standard deviations at candidate points (shape (2000,)).
        f_best: The lowest observed value so far.
        
    Returns:
        Array of scores (shape (2000,)). Higher score indicates higher priority.
    """
    # Expected Improvement for minimization
    # EI(x) = E[max(f_best - f(x), 0)]
    # where f(x) ~ N(mu(x), sigma(x)^2)
    
    # Start time for deadline guard
    start_time = time.time()
    deadline = start_time + 320  # 10s safety margin before 330s limit
    
    # Standardize the improvement
    # z = (f_best - mu) / sigma
    # Handle division by zero: where sigma is very small, we use a small epsilon
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    
    z = (f_best - mu) / sigma_safe
    
    # EI formula:
    # EI = sigma * (z * Phi(z) + phi(z))
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    # Use scipy for normal CDF and PDF if available, otherwise approximate
    try:
        from scipy.stats import norm
        Phi_z = norm.cdf(z)
        phi_z = norm.pdf(z)
    except ImportError:
        # Fallback: use numpy's erf for CDF, manual PDF
        Phi_z = 0.5 * (1 + np.erf(z / np.sqrt(2)))
        phi_z = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    
    # EI = sigma * (z * Phi(z) + phi(z))
    ei = sigma_safe * (z * Phi_z + phi_z)
    
    # EI is already the improvement (higher is better)
    # We want to maximize EI, so return EI directly
    scores = ei
    
    # Handle any potential non-finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
