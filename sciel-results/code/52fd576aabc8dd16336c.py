"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI).
    
    For minimization, EI is defined as:
    EI(x) = E[max(f_best - f(x), 0)]
    
    Given a Gaussian process posterior with mean mu and standard deviation sigma,
    and assuming f(x) ~ N(mu, sigma^2), the EI is:
    EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu) / sigma, Phi is the standard normal CDF,
    and phi is the standard normal PDF.
    
    This function returns the EI score for each candidate. Higher scores
    indicate higher expected improvement, so the candidate with the
    highest score is selected for evaluation.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    # In GP, sigma should be > 0, but numerical errors can occur
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate the z-score
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues in the tails of the normal distribution
    # norm.cdf and norm.pdf are stable for large |z|, but clipping is a safe practice
    # for extreme values that might arise from floating point errors
    z = np.clip(z, -10, 10)
    
    # Calculate Expected Improvement
    # term1: (f_best - mu) * P(f(x) < f_best)
    # term2: sigma * pdf(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure the output is finite
    # If mu or sigma contains NaN/Inf, EI will be NaN/Inf.
    # The problem states "Return 2000 finite scores".
    # We replace non-finite values with 0.0 (neutral score) or a very small negative number.
    # Since EI >= 0, 0.0 is a safe lower bound.
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
