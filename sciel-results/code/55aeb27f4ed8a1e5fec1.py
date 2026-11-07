import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    
    Parameters:
    mu: array of shape (N,) - posterior mean
    sigma: array of shape (N,) - posterior standard deviation
    f_best: float - best observed value (minimum)
    
    Returns:
    scores: array of shape (N,) - acquisition scores (higher is better)
    """
    # Ensure sigma is non-negative and handle zero variance cases
    # If sigma is 0, the distribution is a delta function.
    # EI is max(f_best - mu, 0) if sigma == 0.
    # Using a small epsilon to avoid division by zero in z calculation.
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Calculate z-score
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Calculate EI
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: For minimization, we want to find points where f(x) < f_best.
    # The improvement is f_best - f(x).
    # E[max(f_best - f(x), 0)] = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    # Use scipy.stats.norm for numerical stability
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Handle cases where sigma was originally 0 (or very small)
    # If sigma is effectively 0, EI should be max(f_best - mu, 0)
    # The formula above handles this well if we trust the limit, 
    # but let's explicitly mask where sigma is very small to be safe from numerical artifacts
    # If sigma < 1e-12, the term sigma*phi(z) is negligible, and (f_best-mu)*Phi(z) 
    # approximates max(f_best-mu, 0) if z is large enough.
    # However, if mu > f_best and sigma is tiny, z is large negative, Phi(z) ~ 0, EI ~ 0. Correct.
    # If mu < f_best and sigma is tiny, z is large positive, Phi(z) ~ 1, EI ~ f_best - mu. Correct.
    
    return ei
