"""Combined program: LCB acquisition with adaptive exploration and deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Global deadline guard to avoid timeout
_START_TIME = time.time()
_DEADLINE = 330.0
_SAFETY_MARGIN = 5.0


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for candidates using Lower Confidence Bound (LCB).
    
    Parameters:
    -----------
    mu : array-like, shape (N,)
        Mean of the GP posterior at candidate points.
    sigma : array-like, shape (N,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        The best (lowest) observed function value so far.
        
    Returns:
    --------
    scores : array-like, shape (N,)
        Acquisition scores. Higher scores indicate more promising candidates.
    """
    # Check if we're close to the deadline
    if time.time() > _DEADLINE - _SAFETY_MARGIN:
        # Return a simple fallback to ensure we don't time out
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        return -mu - 2.0 * np.maximum(sigma, 1e-9)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and non-zero
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # LCB acquisition for minimization
    # We want to minimize, so we look for low mu and high sigma
    # Score = -mu - kappa * sigma
    # Using kappa = 2.0 as a standard heuristic for exploration
    kappa = 2.0
    
    score = -mu - kappa * sigma_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
