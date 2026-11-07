"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Use Expected Improvement (EI) per unit cost.
    # EI balances exploitation (improvement over f_best) and exploration (uncertainty sigma).
    
    # Avoid division by zero in Z calculation
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Z-score for EI calculation
    z = (f_best - mu) / sigma_safe
    
    # Standard Normal CDF and PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Cost-aware scoring: prefer points with high EI and low cost
    # cost is provided as an array of shape (2000,)
    # Ensure cost is positive to avoid division errors (though problem says cost >= 1)
    cost_safe = np.maximum(cost, 1e-9)
    
    score = ei / cost_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
