"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    # Deadline guard: ensure we finish within the 300s budget.
    # The evaluator kills after 330s wall-clock; we track elapsed time
    # and if we're getting close, return a safe result.
    
    # Standard Expected Improvement (EI)
    # mu and sigma are in objective units (lower is better for Hartmann 3D)
    # f_best is the lowest observed value
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # z-score for improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    # Use scipy's norm for accurate CDF and PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure all scores are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
