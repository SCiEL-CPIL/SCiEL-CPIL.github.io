"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_DEADLINE = time.time() + 325.0  # 5s safety margin before 330s limit


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of a fixed seeded random RBF mixture in 2D.
    Uses Expected Improvement (EI) for minimization, with a deadline guard.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        GP mean at candidate points.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation at candidate points.
    f_best : float
        Lowest observed value so far.
    
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a simple fallback: exploit the best predicted point
        return -mu.copy()
    
    # Expected Improvement for minimization
    # EI = E[f_best - f(x) | f(x) ~ N(mu(x), sigma(x))]
    # where we want to minimize f.
    # Let z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is CDF and phi is PDF of standard normal.
    
    # Handle sigma near zero to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # Compute EI
    # Phi(z) = norm.cdf(z)
    # phi(z) = norm.pdf(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # EI is non-negative; higher EI means more expected improvement
    # We want to maximize EI, so scores = EI
    scores = ei
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
