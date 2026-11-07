import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Computes the Expected Improvement (EI) acquisition function.
    
    Parameters:
    mu : np.ndarray
        Mean of the Gaussian Process posterior, shape (N,)
    sigma : np.ndarray
        Standard deviation of the Gaussian Process posterior, shape (N,)
    f_best : float
        The best (lowest) observed function value so far.
        
    Returns:
    np.ndarray
        Acquisition scores for each candidate, shape (N,). Higher is better.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate the standardized improvement
    # For minimization, improvement is f_best - f(x)
    # We want to maximize E[f_best - f(x) | f(x) < f_best]
    # Let Z = (f(x) - mu) / sigma ~ N(0, 1)
    # f(x) = mu + sigma * Z
    # Improvement I = max(f_best - (mu + sigma * Z), 0)
    # E[I] = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    # Avoid numerical issues if sigma is very small
    # If sigma is very small, the term (f_best - mu)/sigma might be large.
    # norm.cdf and norm.pdf handle large values reasonably, but let's be safe.
    
    # Add a small exploration constant xi to encourage exploration
    # A small xi helps in early stages. 
    # Since we don't have 't', we can't easily vary xi. 
    # However, EI naturally explores where sigma is high.
    # Let's stick with xi=0 for standard EI, or a very small constant.
    xi = 0.0
    
    diff = f_best - xi - mu
    z = diff / sigma_safe
    
    # Calculate EI
    # Use scipy.stats.norm for CDF and PDF
    # Phi(z) = norm.cdf(z)
    # phi(z) = norm.pdf(z)
    
    # Optimization: 
    # For large negative z, Phi(z) is ~0 and phi(z) is ~0. EI ~ 0.
    # For large positive z, Phi(z) is ~1 and phi(z) is ~0. EI ~ diff.
    
    cdf_vals = norm.cdf(z)
    pdf_vals = norm.pdf(z)
    
    ei = diff * cdf_vals + sigma_safe * pdf_vals
    
    # Ensure finiteness
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
