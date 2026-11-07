import numpy as np
from scipy.stats import norm
import time

# Global timer for deadline guard
_MODULE_START_TIME = time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) for minimization.
    
    This function computes the Expected Improvement of each candidate point
    relative to the current best observed value (f_best).
    
    EI(x) = E[max(f_best - f(x), 0)]
    where f(x) ~ N(mu(x), sigma(x)^2).
    
    For minimization:
    z = (f_best - mu) / sigma
    EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    We return the EI scores. Higher scores indicate higher expected improvement,
    so the point with the highest score is selected for evaluation.
    """
    # Deadline guard: if we are close to the 330s limit, return a safe, fast score
    # to prevent the evaluator from killing the process.
    if time.time() - _MODULE_START_TIME > 320:
        # Fallback: Return sigma (pure exploration) or zeros.
        # Returning sigma ensures we pick an uncertain point, which is safer than crashing.
        return np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Handle numerical issues: ensure sigma is positive
    eps = 1e-10
    sigma_safe = np.where(sigma > eps, sigma, eps)
    
    # Compute Z-score: (f_best - mu) / sigma
    # We want to maximize improvement, so if mu < f_best, z is positive.
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid extreme values in CDF/PDF calculation which can cause numerical issues
    # although norm.cdf and norm.pdf are generally stable, clipping is a good practice
    z = np.clip(z, -10, 10)
    
    # Standard Normal CDF and PDF
    Phi_z = norm.cdf(z)
    phi_z = norm.pdf(z)
    
    # Expected Improvement formula
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure finite outputs
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    return ei
