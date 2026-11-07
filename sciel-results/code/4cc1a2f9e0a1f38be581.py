"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import erfc
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6D.
    
    Uses a hybrid of Expected Improvement (EI) with a small exploration bonus.
    The EI term handles exploitation while the exploration bonus (0.1 * sigma)
    helps escape local minima in the high-dimensional landscape with limited samples.
    
    Parameters:
    -----------
    mu : array-like, shape (N,)
        Posterior mean at candidate points (in objective units)
    sigma : array-like, shape (N,)
        Posterior standard deviation at candidate points (in objective units)
    f_best : float
        Best (lowest) observed value so far
    
    Returns:
    --------
    scores : array, shape (N,)
        Acquisition scores; higher means more promising to evaluate next
    """
    t0 = time.time()
    deadline = t0 + 280.0  # Safety margin before 300s limit
    
    # Ensure inputs are proper numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle non-finite inputs gracefully
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma))):
        return np.arange(len(mu), dtype=np.float64)
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-8)
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma
    
    # Compute Phi(z) = 0.5 * erfc(-z / sqrt(2))
    # Using erfc for numerical stability
    sqrt2 = np.sqrt(2.0)
    phi_z = 0.5 * erfc(-z / sqrt2)
    
    # Compute phi(z) = (1 / sqrt(2*pi)) * exp(-z^2 / 2)
    phi_pdf = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * phi_z + sigma * phi_pdf
    
    # Add a small exploration term to prevent pure exploitation
    # This helps escape local minima in the 6D Hartmann landscape
    exploration_term = 0.1 * sigma
    
    scores = ei + exploration_term
    
    # Ensure all returned scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline (should be very fast, O(N) computation)
    if time.time() > deadline:
        # Return current scores as fallback
        pass
    
    return scores
# EVOLVE-BLOCK-END
