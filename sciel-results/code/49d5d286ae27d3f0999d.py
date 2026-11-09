# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of ackley 2d.
    
    Uses pure Expected Improvement (EI) acquisition function.
    EI is a standard and robust choice for Bayesian optimization,
    particularly effective in low-dimensional spaces like 2D.
    It directly optimizes the expected reduction in the best observed value.
    """
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate the Z-score
    # z = (f_best - mu) / sigma
    # If mu < f_best, z is positive.
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical instability in norm.cdf/pdf
    # norm.cdf(10) is effectively 1, norm.pdf(10) is effectively 0
    z_clipped = np.clip(z, -10, 10)
    
    # Expected Improvement formula:
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is CDF and phi is PDF of standard normal
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    
    # EI is non-negative by definition, but due to numerical errors it might be tiny negative
    ei = np.maximum(ei, 0)
    
    # Ensure all scores are finite
    # If sigma is 0 and mu == f_best, z is 0, ei is 0.
    # If sigma is 0 and mu < f_best, z is inf, cdf is 1, pdf is 0, ei is f_best - mu.
    # If sigma is 0 and mu > f_best, z is -inf, cdf is 0, pdf is 0, ei is 0.
    
    score = ei
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
