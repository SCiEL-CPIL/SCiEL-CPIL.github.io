"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    
    EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x)) / sigma(x)) + sigma(x) * phi((f_best - mu(x)) / sigma(x))
    
    Parameters:
    -----------
    mu : array_like, shape (n,)
        Mean of the GP posterior.
    sigma : array_like, shape (n,)
        Standard deviation of the GP posterior.
    f_best : float
        Best observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        EI scores. Higher means evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Sanitize inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-12)
    
    # Calculate Expected Improvement
    # For minimization, we want to find points where f(x) < f_best.
    # Improvement is max(0, f_best - f(x)).
    # Since f(x) ~ N(mu, sigma^2), E[max(0, f_best - f(x))] is the EI formula.
    
    delta = f_best - mu
    z = delta / sigma
    
    # Avoid division by zero or numerical issues
    # If sigma is very small, z is large.
    # Phi(z) -> 1, phi(z) -> 0.
    # EI -> delta * 1 + 0 = f_best - mu.
    
    # Use scipy for accurate normal CDF and PDF
    # norm.cdf(z) is Phi(z)
    # norm.pdf(z) is phi(z)
    
    # To avoid overflow in exp for very negative z (though z is delta/sigma, 
    # if mu > f_best, z is negative. If mu << f_best, z is positive).
    # Actually, if mu > f_best, delta < 0, z < 0. EI should be 0.
    # The formula handles this naturally as Phi(z) is small and phi(z) is small,
    # but let's be careful.
    
    # Standard stable implementation:
    # EI = delta * Phi(z) + sigma * phi(z)
    
    # However, if delta < 0 (mu > f_best), the improvement is 0.
    # The formula (f_best - mu) * Phi(z) + sigma * phi(z) yields 0 when mu > f_best?
    # Let's check: if mu = f_best + eps, z = -eps/sigma.
    # Phi(-eps/sigma) is small. phi(-eps/sigma) is small.
    # It approaches 0.
    
    # We can simply compute it.
    scores = delta * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return scores
# EVOLVE-BLOCK-END
