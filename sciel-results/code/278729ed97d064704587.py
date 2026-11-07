"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Ensure sigma is strictly positive for numerical stability
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma
    
    # Standard Expected Improvement for minimization
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    ei = (f_best - mu) * cdf + sigma * pdf
    
    # Clip EI to a small positive value to ensure log is defined and finite
    # This prevents -inf scores for points with zero expected improvement
    ei = np.clip(ei, 1e-12, None)
    
    # Use Log-Expected Improvement
    # This provides a more robust scaling of the EI values,
    # reducing the dominance of outliers with extremely high EI
    # and maintaining better discrimination in the mid-range.
    scores = np.log(ei)
    
    # Ensure all scores are finite
    # Replace any remaining non-finite values with a very low score
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
