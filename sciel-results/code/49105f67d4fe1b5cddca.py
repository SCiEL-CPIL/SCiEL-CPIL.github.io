"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        GP mean predictions.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed objective value.
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Deadline guard: track elapsed time and return best result found
    # with a safety margin before the 330s limit.
    start_time = time.time()
    deadline = 330.0 - 5.0  # 5 second safety margin
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Handle edge cases
    if n == 0:
        return np.array([])
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute expected improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Use numerically stable computation
    # For large positive z, Phi(z) ~ 1, phi(z) ~ 0
    # For large negative z, Phi(z) ~ 0, phi(z) ~ 0
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -10, 10)
    
    cdf = norm.cdf(z_clipped)
    pdf = norm.pdf(z_clipped)
    
    ei = (f_best - mu) * cdf + sigma * pdf
    
    # Ensure EI is non-negative (can be slightly negative due to numerical errors)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scoring: EI per unit cost
    # score = EI / cost
    # This prioritizes high-gain, low-cost evaluations
    scores = ei / np.maximum(cost, 1e-9)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Return current result if we're over deadline
        return scores
    
    return scores
# EVOLVE-BLOCK-END
