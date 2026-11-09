"""
Acquisition function for Bayesian optimization of sphere 1d.

This implementation uses Expected Improvement (EI) for minimization.
EI balances exploration (uncertainty) and exploitation (current best)
by quantifying the expected reduction in the objective function value.
"""
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Compute Expected Improvement scores for the given posterior parameters.
    
    Parameters:
    mu: np.ndarray of shape (N,), posterior mean.
    sigma: np.ndarray of shape (N,), posterior standard deviation.
    f_best: float, the best observed value so far.
    
    Returns:
    np.ndarray of shape (N,), scores where higher is better.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-8)
    
    # Ensure f_best is a scalar float
    f_best = float(f_best)
    
    # Standard Normal Z-score
    # Z = (f_best - mu) / sigma
    # If sigma is very small, Z becomes very large positive or negative.
    # We use np.divide with where to handle zeros safely, though sigma is clamped.
    z = (f_best - mu) / sigma
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is CDF and phi is PDF of standard normal.
    
    phi_z = norm.pdf(z)
    phi_z = np.where(np.isfinite(phi_z), phi_z, 0.0)
    
    cdf_z = norm.cdf(z)
    cdf_z = np.where(np.isfinite(cdf_z), cdf_z, 0.0)
    
    improvement = f_best - mu
    # If improvement is negative (mu < f_best is not possible since f_best is min, 
    # but numerically mu could be slightly lower if GP is noisy/overfit? No, f_best is observed min.
    # However, GP mean can be lower than f_best. If mu < f_best, improvement > 0.
    # If mu > f_best, improvement < 0. EI should be 0 in this case theoretically, 
    # but the formula handles it: 
    # If z < 0, Phi(z) is small, phi(z) is small. 
    # Actually, standard EI formula is max(0, f_best - mu). The analytic form assumes 
    # we are integrating the part where f_best - mu > 0.
    # The formula EI = (f_best - mu)Phi(z) + sigma*phi(z) is valid for all z.
    # If z is very negative (mu >> f_best), Phi(z) ~ 0, phi(z) ~ 0, EI ~ 0.
    # If z is very positive (mu << f_best), Phi(z) ~ 1, EI ~ f_best - mu.
    
    ei = improvement * cdf_z + sigma * phi_z
    
    # Clean up any non-finite values
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # EI is non-negative. Higher is better.
    return ei
