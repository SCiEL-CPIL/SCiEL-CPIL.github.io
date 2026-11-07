"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) for minimization.
    
    mu: (2000,) array of GP mean predictions
    sigma: (2000,) array of GP standard deviations
    f_best: lowest observed value so far
    
    Returns: (2000,) array of acquisition scores (higher = more promising)
    """
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate the improvement over f_best
    # Since we are minimizing, improvement is f_best - mu
    improvement = f_best - mu
    
    # Standardize the improvement
    z = improvement / sigma
    
    # Expected Improvement formula for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is CDF and phi is PDF of standard normal
    
    # Use scipy.stats.norm for stable computation
    pdf = norm.pdf(z)
    cdf = norm.cdf(z)
    
    score = improvement * cdf + sigma * pdf
    
    # Ensure all scores are finite
    # EI is theoretically non-negative, but numerical errors might cause slight negatives or inf/nan
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
