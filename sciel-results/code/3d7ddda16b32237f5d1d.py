"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Ackley 2D.
    
    Uses Expected Improvement (EI) as the primary acquisition function.
    EI naturally balances exploration and exploitation:
    - When f_best is large (early), points with low mu or high sigma get high EI
    - When f_best is small (late), EI focuses on points likely to improve
    
    Numerical stability is ensured with safe sigma clamping and finite checks.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    sigma = np.maximum(sigma, 1e-10)
    
    f_best = float(f_best)
    if not np.isfinite(f_best):
        f_best = 1.0
    
    # Compute Expected Improvement (EI) for minimization
    # Improvement = max(0, f_best - f(x))
    # E[Improvement] = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Use standard normal CDF and PDF
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure non-negative and finite
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
