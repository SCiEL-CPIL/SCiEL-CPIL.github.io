"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import t


def acquisition(mu, sigma, f_best):
    # Student's t-distribution based Expected Improvement
    # Using 5 degrees of freedom to account for heavy-tailed uncertainty
    # in the limited sample budget.
    
    nu = 5.0
    
    # Safe sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Z-score
    z = (f_best - mu) / sigma_safe
    
    # Compute t-CDF and t-PDF
    t_cdf = t.cdf(z, nu)
    t_pdf = t.pdf(z, nu)
    
    # Coefficient for the PDF term in the t-EI approximation
    coeff = (nu + 1.0) / (nu - 1.0)
    
    # Expected Improvement
    # EI_t = (f_best - mu) * T(z) + sigma * coeff * t_pdf(z)
    scores = (f_best - mu) * t_cdf + sigma * coeff * t_pdf
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
