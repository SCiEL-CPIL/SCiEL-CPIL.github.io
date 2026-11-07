"""Acquisition function for Bayesian optimization of Hartmann 3D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Compute a robust acquisition score for 2000 candidate points.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions at candidates.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions at candidates.
    f_best : float
        Best (lowest) observed function value so far.
    
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Acquisition scores; higher is better.
    """
    # Deadline guard: ensure we return within time limits
    start_time = time.time()
    deadline = start_time + 295.0  # Safety margin before 300s limit
    
    # Ensure inputs are proper arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-12)
    
    # Compute z-scores for expected improvement
    # z = (f_best - mu) / sigma
    # When mu < f_best, z > 0 (potential improvement)
    # When mu > f_best, z < 0 (unlikely improvement)
    
    # Handle potential overflow/underflow in z computation
    z = (f_best - mu) / sigma
    z = np.clip(z, -100, 100)  # Prevent extreme values
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is standard normal CDF and phi is standard normal PDF
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # EI formula
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Combine EI with a small exploration bonus based on sigma
    # This helps balance exploitation and exploration
    # Use a temperature-like scaling
    exploration_weight = 0.1
    exploration_bonus = exploration_weight * sigma
    
    # Final score: EI + exploration bonus
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        # If we're close to the deadline, just return what we have
        pass
    
    return scores
# EVOLVE-BLOCK-END
