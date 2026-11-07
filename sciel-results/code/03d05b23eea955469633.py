"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Deterministic Lower Confidence Bound (LCB) with adaptive exploration.
    
    Uses a high exploration coefficient to target regions of high posterior
    uncertainty, which is critical for escaping local optima in the multimodal
    Hartmann 6D landscape within a tight 30-evaluation budget.
    
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
    
    # Use a high exploration coefficient for aggressive exploration
    # This helps escape local optima in the multimodal landscape
    kappa = 3.0
    
    # LCB acquisition: we want to minimize, so we score as negative LCB
    # LCB = mu - kappa * sigma (lower is better for minimization)
    # Score = -LCB = -mu + kappa * sigma (higher is better for acquisition)
    scores = -mu + kappa * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
