"""Acquisition function for Bayesian optimization of Weierstrass 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Compute Expected Improvement (EI) acquisition scores for minimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        The best (lowest) objective value observed so far.
        
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores. Higher is better.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Standardized improvement
    # For minimization, we want to minimize f(x).
    # Improvement is max(f_best - f(x), 0).
    # Let Z = (f(x) - f_best) / sigma.
    # If f(x) < f_best, Z is negative.
    # EI = E[max(f_best - f(x), 0)]
    #    = E[(f_best - f(x)) * 1_{f(x) < f_best}]
    #    = (f_best - mu) * P(f(x) < f_best) + sigma * phi(Z)
    # where Z = (f(x) - f_best) / sigma
    # P(f(x) < f_best) = P(Z < (f_best - mu)/sigma) = Phi((f_best - mu)/sigma)
    # Let Z' = (mu - f_best) / sigma. Then (f_best - mu)/sigma = -Z'.
    # P(f(x) < f_best) = Phi(-Z') = 1 - Phi(Z')
    # EI = (f_best - mu) * (1 - Phi(Z')) + sigma * phi(Z')
    # Note: phi is symmetric, phi(Z') = phi(-Z')
    
    # To avoid numerical issues with large values, we can use the standard formula:
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * pdf((f_best - mu)/sigma)
    
    z = (f_best - mu) / sigma_safe
    # Phi is the CDF of the standard normal distribution
    # pdf is the PDF of the standard normal distribution
    
    # Use scipy.stats.norm for accurate calculation
    phi = norm.cdf(z)
    pdf = norm.pdf(z)
    
    ei = (f_best - mu) * phi + sigma_safe * pdf
    
    # Ensure non-negative EI (numerical precision might cause small negatives)
    ei = np.maximum(ei, 0.0)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(ei), ei, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
