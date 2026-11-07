import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement acquisition function for Bayesian optimization of Ackley 2D.
    
    Computes EI(mu, sigma, f_best) and multiplies by (1.0 / cost) to get
    expected improvement per unit cost. This prioritizes high-gain, low-cost evaluations.
    """
    # Ensure inputs are 1D arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: clamp sigma to be non-negative and finite
    sigma = np.clip(sigma, 0.0, None)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-6)
    
    # Compute Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # Phi is the standard normal CDF, phi is the standard normal PDF
    
    # Handle sigma = 0 case: EI = max(f_best - mu, 0)
    eps = 1e-10
    safe_sigma = np.maximum(sigma, eps)
    
    z = (f_best - mu) / safe_sigma
    
    # Standard normal CDF and PDF
    # Using scipy.special for accuracy
    try:
        from scipy.special import ndtr, ndtri
        # ndtr is the standard normal CDF
        # pdf = exp(-0.5 * z^2) / sqrt(2*pi)
        phi_z = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
        Phi_z = ndtr(z)
    except ImportError:
        # Fallback: use numpy's erf for CDF approximation
        # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
        Phi_z = 0.5 * (1.0 + np.erf(z / np.sqrt(2.0)))
        phi_z = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    
    # Expected Improvement
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scoring: EI per unit cost
    score = ei / cost
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
