"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using dynamic UCB that shifts from exploration to
    exploitation over the course of optimization.
    
    Parameters
    ----------
    mu : array_like, shape (n,)
        Posterior mean at candidate points.
    sigma : array_like, shape (n,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best (lowest) observed function value so far.

    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Guard against non-finite values
    mu = np.where(np.isfinite(mu), mu, np.nanmax(mu) if np.any(np.isfinite(mu)) else 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma = np.maximum(sigma, 1e-12)
    
    # Estimate progress based on f_best relative to the range of mu
    # Early in optimization, f_best is likely far from the minimum
    # As we find better points, f_best approaches the minimum
    
    # Use a heuristic: if f_best is very low relative to mu's scale, we're late
    mu_scale = np.std(mu) + 1e-12
    f_best_norm = f_best / (abs(np.mean(mu)) + mu_scale + 1e-12)
    
    # Progress estimate: 0 at start, 1 near convergence
    # If f_best is much smaller than typical mu values, we're progressing well
    progress = np.clip(1.0 - abs(f_best_norm) / 10.0, 0.0, 1.0)
    
    # Dynamic beta: start with high exploration (beta ~ 3.0), decrease to low (beta ~ 0.5)
    beta = 3.0 - 2.5 * progress
    
    # Dynamic UCB: -mu + beta * sigma
    # This favors low mean (exploitation) and high uncertainty (exploration)
    scores = -mu + beta * sigma
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
