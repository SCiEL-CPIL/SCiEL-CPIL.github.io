"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Use Normalized Expected Improvement (NEI)
    # NEI is equivalent to ranking by z-score, which is robust to the scale of f_best
    # and provides better exploration/exploitation balance in early stages.
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-scores
    z = (f_best - mu) / sigma_safe
    
    # Calculate Normalized EI: z * Phi(z) + phi(z)
    # This is monotonic with z, so it effectively ranks by Probability of Improvement
    # but with a different scaling that may be numerically more stable or preferred
    # by the optimization dynamics in this specific task context.
    
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    scores = z * cdf_z + pdf_z
    
    # Handle any potential numerical issues (though z, cdf, pdf should be finite)
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
