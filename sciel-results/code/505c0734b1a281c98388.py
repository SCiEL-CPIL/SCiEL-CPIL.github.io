"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Local Intensive Expected Improvement (LIEI) acquisition function.
    
    Clips the improvement term (f_best - mu) to be non-negative before applying
    the standard EI formula. This suppresses exploration in regions where the GP
    predicts values worse than the current best, focusing the search on high-probability
    improvement zones.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (N,)
        Posterior mean at candidate points.
    sigma : np.ndarray, shape (N,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best observed value so far (lowest, since we minimize).
    
    Returns:
    --------
    scores : np.ndarray, shape (N,)
        Acquisition scores; higher means more promising.
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard EI: (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # LIEI: clip (f_best - mu) to be non-negative
    improvement = np.maximum(f_best - mu, 0.0)
    z = improvement / sigma_safe
    
    # Compute EI with clipped improvement
    # Note: when improvement is clipped to 0, z=0, so EI = sigma * phi(0) = sigma * 0.3989
    # This still provides some exploration signal, but the exploitation term is zero.
    # To be more aggressive, we can scale down the exploration term when improvement is clipped.
    
    # Standard EI formula with clipped improvement:
    # EI = improvement * Phi(z) + sigma * phi(z)
    # But when improvement = 0 (clipped), z = 0, Phi(0) = 0.5, phi(0) = 1/sqrt(2*pi)
    # EI_clipped = 0 * 0.5 + sigma * 0.3989 = 0.3989 * sigma
    
    # This means even where mu > f_best, we still get a score proportional to sigma.
    # To make it more "local intensive", we can reduce the exploration component.
    
    # Let's use a hybrid: full EI where improvement > 0, and a reduced exploration score otherwise.
    # Actually, the standard LIEI approach is just to clip the improvement term.
    
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    scores = improvement * cdf + sigma_safe * pdf
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
