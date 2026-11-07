"""Acquisition function using Expected Improvement (EI)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function.
    
    EI(x) = (f_best - mu(x)) * Phi(Z) + sigma(x) * phi(Z)
    where Z = (f_best - mu(x)) / sigma(x)
    
    We want to minimize the function, so we look for points where the
    expected improvement over the current best (f_best) is high.
    Higher EI score means more likely to find a new minimum.
    """
    n = len(mu)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Calculate Z-score: how many standard deviations f_best is above mu
    # If mu < f_best, Z is positive. If mu > f_best, Z is negative.
    Z = (f_best - mu) / sigma_safe
    
    # Calculate EI
    # Phi(Z) is the CDF of the standard normal distribution
    # phi(Z) is the PDF of the standard normal distribution
    # We use scipy.stats.norm for these
    phi_Z = norm.pdf(Z)
    Phi_Z = norm.cdf(Z)
    
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # Note: (f_best - mu) is the potential improvement if we are at the mean.
    # If mu > f_best, (f_best - mu) is negative, but Phi(Z) is small (since Z is negative),
    # and the second term dominates slightly, leading to small positive EI (exploration).
    # If mu < f_best, (f_best - mu) is positive, and Phi(Z) is large, leading to high EI.
    
    EI = (f_best - mu) * Phi_Z + sigma_safe * phi_Z
    
    # Ensure finite scores
    EI = np.where(np.isfinite(EI), EI, 0.0)
    
    return EI
# EVOLVE-BLOCK-END
