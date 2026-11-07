import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Compute Expected Improvement (EI) for minimization.
    
    EI(x) = E[max(f_best - f(x), 0)]
    For f(x) ~ N(mu, sigma^2):
    EI(x) = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is positive to avoid division by zero issues in z-score calculation
    # However, if sigma is 0, the distribution is deterministic.
    # We handle this by masking.
    
    # Calculate the z-score
    # z = (f_best - mu) / sigma
    # To avoid division by zero, we use a small epsilon
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    z = (f_best - mu) / sigma_safe
    
    # Calculate EI
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # For points where sigma was very small (essentially 0):
    # If mu < f_best, the improvement is exactly f_best - mu.
    # If mu >= f_best, the improvement is 0.
    # The formula above with eps approximates this, but let's be precise for numerical stability.
    # Actually, if sigma -> 0, z -> +/- inf.
    # If mu < f_best, z -> +inf. Phi(z) -> 1, phi(z) -> 0. EI -> f_best - mu.
    # If mu > f_best, z -> -inf. Phi(z) -> 0, phi(z) -> 0. EI -> 0.
    # So the formula is stable even with eps, provided we don't get NaNs.
    
    # Ensure finite values
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # EI is always non-negative. Higher is better.
    return ei
