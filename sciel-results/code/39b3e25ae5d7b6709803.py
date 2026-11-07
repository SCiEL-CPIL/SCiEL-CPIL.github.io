"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6D.
    
    Uses Expected Improvement (EI) with a small exploration bonus.
    The EI term handles exploitation while the exploration bonus (0.1 * sigma)
    helps escape local minima in the high-dimensional landscape with limited samples.
    """
    # Track time to ensure we finish well within the 300s limit
    t0 = time.time()
    
    # Ensure inputs are proper numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle non-finite inputs gracefully
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma))):
        return np.zeros(len(mu), dtype=np.float64)
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-8)
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma
    
    # Compute Phi(z) and phi(z) using scipy.stats for numerical stability
    phi_z = norm.cdf(z)
    phi_pdf = norm.pdf(z)
    
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
    elapsed = time.time() - t0
    if elapsed > 280.0:
        # Return current scores as fallback
        pass
    
    return scores
# EVOLVE-BLOCK-END
