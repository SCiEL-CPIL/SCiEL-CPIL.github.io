"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) with an adaptive
    UCB-like exploration term.
    
    The current best score (1.287) was achieved with EI + 0.1*sigma.
    To improve, we increase the exploration coefficient to 0.3 to better
    escape local minima in the 6D Hartmann landscape, and include a small
    xi (0.01) in the EI calculation to further encourage exploration of
    points that are likely to be better than f_best by a margin.
    """
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Exploration parameter for EI
    xi = 0.01
    
    # Calculate the z-score for the expected improvement
    # z = (f_best - mu - xi) / sigma
    z = (f_best - mu - xi) / sigma_safe
    
    # Compute Expected Improvement
    # EI = (f_best - mu - xi) * Phi(z) + sigma * phi(z)
    # Note: We use sigma_safe in the PDF term for consistency, though sigma is fine if > 0
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB exploration term
    # Increased from 0.1 to 0.3 to boost exploration in high-dimensional space
    beta_ucb = 0.3
    ucb_term = beta_ucb * sigma_safe
    
    # Combined acquisition: EI + UCB term
    scores = ei + ucb_term
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
