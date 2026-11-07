"""Acquisition function for Bayesian optimization of a 2D RBF mixture."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 325.0  # Safety margin

def _time_remaining():
    return _DEADLINE - time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) for minimization
    with an exploration bonus based on posterior uncertainty.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP posterior mean at candidate points.
    sigma : ndarray, shape (2000,)
        GP posterior std at candidate points.
    f_best : float
        Best (lowest) observed objective value so far.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising points.
    """
    # Check deadline
    if _time_remaining() < 0.5:
        # Fallback: simple PI calculation
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        return norm.cdf(z)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Add a small exploration bonus to encourage visiting uncertain regions
    # This helps escape local minima in multi-modal landscapes
    # Scale the bonus by the relative uncertainty
    exploration_bonus = 0.1 * sigma_safe / (np.std(sigma) + 1e-9)
    
    scores = ei + exploration_bonus
    
    # Ensure finite output
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
