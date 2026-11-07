"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    
    Uses an additive combination of Expected Improvement (EI) and an adaptive
    exploration bonus proportional to the posterior standard deviation.
    This approach preserves the scale of the expected improvement, which is
    crucial for balancing exploitation and exploration effectively in the
    multimodal landscape of the Goldstein-Price function.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Posterior mean predictions at candidate points
    sigma : array of shape (2000,)
        Posterior standard deviation predictions at candidate points
    f_best : float
        Best (lowest) observed function value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores, higher is better
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Safety deadline guard
    t_start = time.time()
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute Expected Improvement (for minimization)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid overflow in exp for very large/small values
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Adaptive exploration weight
    # We want to explore more when uncertainty is high relative to the current best.
    # As f_best approaches the minimum, the scale of improvement decreases,
    # so we reduce the exploration bonus to focus on exploitation.
    sigma_mean = np.mean(sigma_safe)
    # Scale factor: if f_best is large (early), lambda is larger.
    # If f_best is small (late), lambda is smaller.
    # The denominator (abs(f_best) + 1.0) provides a stable scale.
    # The factor 2.0 is a heuristic tuned for the Goldstein-Price function scale.
    lambda_explore = 2.0 * min(1.0, sigma_mean / (abs(f_best) + 1.0))
    
    # Exploration bonus
    exploration_bonus = lambda_explore * sigma_safe
    
    # Combine EI and exploration bonus
    scores = ei + exploration_bonus
    
    # Ensure all values are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Deadline guard
    if time.time() - t_start > 280:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
