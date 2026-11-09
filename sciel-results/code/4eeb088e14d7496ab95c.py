"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) Acquisition Function for Minimization.
    
    This strategy uses the standard EI criterion, which calculates the expected
    amount by which a candidate point would improve upon the current best value.
    It provides a smooth, deterministic score that balances exploration (high
    uncertainty) and exploitation (low predicted mean).
    
    EI(x) = (f_best - mu(x)) * Phi(Z) + sigma(x) * phi(Z)
    where Z = (f_best - mu(x)) / sigma(x)
    
    Higher EI means higher probability of finding a new minimum.
    """
    n = len(mu)
    
    # Ensure sigma is valid to avoid division by zero and log issues
    # A very small epsilon prevents numerical instability in the normal PDF
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate Z-score: how many standard deviations below f_best is the mean?
    # If mu < f_best, Z is positive.
    Z = (f_best - mu) / sigma_safe
    
    # Phi(Z) is the CDF of the standard normal distribution
    # phi(Z) is the PDF of the standard normal distribution
    Phi_Z = norm.cdf(Z)
    phi_Z = norm.pdf(Z)
    
    # Calculate Expected Improvement
    # Term 1: Expected improvement if mu < f_best
    # Term 2: Correction for uncertainty
    ei = (f_best - mu) * Phi_Z + sigma_safe * phi_Z
    
    # Ensure all scores are finite
    # EI should be non-negative, but numerical errors might cause tiny negatives
    # We clip to 0 to be safe, though EI is theoretically >= 0
    ei = np.maximum(ei, 0.0)
    
    # Handle any potential NaNs or Infs (shouldn't happen with safe sigma, but just in case)
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e6, neginf=0.0)
    
    return ei
# EVOLVE-BLOCK-END
