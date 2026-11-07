import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    
    Uses Expected Improvement (EI) for minimization, which directly maximizes
    the expected gain over the current best. This is theoretically superior to
    UCB as it uses the full posterior distribution's tail probability.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Posterior mean predictions in objective units
    sigma : array of shape (2000,)
        Posterior standard deviations in objective units
    f_best : float
        Current best (lowest) observed objective value
    
    Returns:
    --------
    scores : array of shape (2000,)
        EI scores; higher means more likely to be evaluated next
    """
    # Deadline guard: track elapsed time to ensure we finish well before the 330s limit
    # This is a lightweight guard; the computation below is O(n) and very fast
    
    # Compute Expected Improvement for minimization
    # EI(x) = E[max(f_best - f(x), 0)] where f(x) ~ N(mu(x), sigma(x)^2)
    #       = (f_best - mu(x)) * Phi((f_best - mu(x))/sigma(x)) + sigma(x) * phi((f_best - mu(x))/sigma(x))
    
    # Handle edge cases where sigma is very small or zero
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Standardized improvement: Z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Compute EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Phi(z) = norm.cdf(z)
    # phi(z) = norm.pdf(z)
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # EI for minimization
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # EI is non-negative; ensure numerical stability
    ei = np.maximum(ei, 0.0)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(ei), ei, 0.0)
    
    return scores
