"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    
    Uses Expected Improvement (EI) with adaptive xi parameter.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        Mean of the GP posterior at candidate points
    sigma : np.ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points
    f_best : float
        Best (lowest) observed function value so far
    
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Acquisition scores; higher means more likely to be evaluated next
    """
    t_start = time.time()
    
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-10)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-10)
    mu_safe = np.where(np.isfinite(mu), mu, 0.0)
    
    if f_best is None or not np.isfinite(f_best):
        f_best = np.min(mu_safe)
    
    # Adaptive xi: scales with uncertainty to balance exploration/exploitation
    # When sigma is large, xi is larger to encourage exploration
    sigma_mean = np.mean(sigma_safe)
    xi = 0.1 * abs(f_best) + 0.05 * sigma_mean
    
    # Compute EI for minimization
    # EI(x) = (f_best - mu(x) - xi) * Phi((f_best - mu(x) - xi) / sigma(x)) 
    #        + sigma(x) * phi((f_best - mu(x) - xi) / sigma(x))
    # where Phi is CDF and phi is PDF of standard normal
    
    z = (f_best - mu_safe - xi) / sigma_safe
    ei = (f_best - mu_safe - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # EI is always non-negative, higher is better
    scores = np.maximum(ei, 0.0)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Safety check: if we're taking too long, return early
    if time.time() - t_start > 10.0:
        # Fallback: just use pure exploitation
        scores = -mu_safe
    
    return scores
# EVOLVE-BLOCK-END
