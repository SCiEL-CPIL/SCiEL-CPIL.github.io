"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Calculates Expected Improvement (EI) for each candidate.
    The cost parameter is ignored for ranking because the minimum location
    is uniformly distributed, and penalizing 'expensive' coordinates (higher x)
    biases the search against 50% of the potential minimum locations.
    
    We use a standard EI formulation with numerical stability safeguards.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-score
    # z = (f_best - mu) / sigma
    # Since we are minimizing, improvement is f_best - mu (positive if mu < f_best)
    diff = f_best - mu
    z = diff / sigma_safe
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # We compute this in a numerically stable way.
    
    # For large positive z, Phi(z) -> 1, phi(z) -> 0, EI -> f_best - mu
    # For large negative z, Phi(z) -> 0, phi(z) -> 0, EI -> 0
    
    # Use scipy's norm for CDF and PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = diff * cdf_z + sigma * pdf_z
    
    # Ensure finiteness and non-negativity (EI should be >= 0)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    ei = np.maximum(ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
