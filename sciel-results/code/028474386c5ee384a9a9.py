"""Acquisition function for Bayesian optimization of sphere 1d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Use Expected Improvement (EI) for minimization.
    
    EI calculates the expected improvement over the current best value f_best.
    For a point with mean mu and std sigma, the expected improvement is:
    EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    
    This approach naturally balances exploration and exploitation without
    requiring manual tuning of exploration coefficients. It is a standard
    Bayesian optimization acquisition function.
    """
    # Clamp sigma to avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    
    # Calculate the normalized improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Compute EI
    # term1 = (f_best - mu) * Phi(z)
    # term2 = sigma * phi(z)
    # EI = term1 + term2
    
    # Use scipy.stats.norm for CDF and PDF
    # Note: norm.pdf is the standard normal PDF phi(z)
    # Note: norm.cdf is the standard normal CDF Phi(z)
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure all scores are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
