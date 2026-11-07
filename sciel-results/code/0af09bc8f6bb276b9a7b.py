"""Acquisition function for Bayesian optimization of Branin 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Refined hybrid acquisition: EI with a calibrated exploration term.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean at candidate points.
    sigma : ndarray, shape (2000,)
        GP standard deviation at candidate points.
    f_best : float
        Best (lowest) observed function value.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising points.
    """
    sigma = np.maximum(sigma, 1e-12)
    
    # Core Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Exploration bonus: encourage visiting high-uncertainty regions
    # Use a small multiple of sigma to avoid dominating EI
    # Scale by how much we've improved from initial guess
    # Branin initial random points typically give f ~ 30-100, optimum is 0.398
    # Early: f_best ~ 50+, explore more. Late: f_best ~ 1, exploit more.
    f_ratio = np.clip(f_best / 10.0, 0.0, 1.0)  # 1.0 when f_best >= 10, 0.0 when f_best <= 0
    beta = 0.1 + 0.4 * f_ratio  # 0.5 early, 0.1 late
    
    exploration = beta * sigma
    
    scores = ei + exploration
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
