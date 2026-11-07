"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: track start time for safety
_start_time = time.time()
_DEADLINE = 300.0  # seconds, with margin before the 330s kill


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for Bayesian optimization.
    
    Uses EI which naturally balances exploration and exploitation by
    considering the probability of improvement weighted by the magnitude
    of improvement. This is more robust than LCB for multimodal functions
    like Hartmann 6D within tight evaluation budgets.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Mean of the GP posterior at candidate points
    sigma : array of shape (2000,)
        Standard deviation of the GP posterior at candidate points
    f_best : float
        Best (lowest) observed function value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores; higher means evaluate next
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle edge cases
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):
        # Fallback: use uniform random scores if inputs are invalid
        return np.random.randn(len(mu))
    
    # Clip sigma to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Deadline guard: if we're running out of time, return a simple score
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Return a simple exploitation-based score to be safe
        scores = -mu
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    # Compute the standardized improvement
    improvement = f_best - mu  # positive when mu < f_best (potential improvement)
    z = improvement / sigma
    
    # Compute the EI components
    # Phi(z) = standard normal CDF
    # phi(z) = standard normal PDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # EI = improvement * Phi(z) + sigma * phi(z)
    # For minimization, we want to maximize the expected reduction from f_best
    scores = improvement * Phi_z + sigma * phi_z
    
    # Ensure all scores are finite and non-negative
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
