import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 285

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI).
    EI balances exploitation (improving the best value) and exploration (reducing uncertainty)
    by maximizing the expected improvement over the current best.
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Basic safety checks
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):
        return np.zeros_like(mu)
    
    # Handle cases where sigma is very small to avoid division by zero
    # In EI, if sigma is 0, the improvement is deterministic: max(f_best - mu, 0)
    # But numerically, we use a small epsilon.
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues in norm.cdf/pdf
    z = np.clip(z, -50, 50)
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is CDF and phi is PDF of standard normal
    phi_z = norm.pdf(z)
    phi_z = np.where(np.isfinite(phi_z), phi_z, 0.0)
    
    cdf_z = norm.cdf(z)
    cdf_z = np.where(np.isfinite(cdf_z), cdf_z, 0.0)
    
    improvement = f_best - mu
    # If mu > f_best, improvement is negative, but EI should be 0 in that case?
    # The formula handles this naturally because (f_best - mu) is negative, but Phi(z) is small.
    # Actually, if mu > f_best, z is negative.
    # Let's ensure EI is non-negative.
    ei_scores = improvement * cdf_z + sigma_safe * phi_z
    
    # EI should be >= 0
    ei_scores = np.maximum(ei_scores, 0.0)
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(ei_scores), ei_scores, 0.0)
    
    return scores
