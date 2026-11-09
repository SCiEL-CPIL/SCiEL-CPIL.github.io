"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Knowledge Gradient (KG) acquisition function for minimization.
    KG(x) = f_best - E[min(f_best, f(x))]
    For a Gaussian posterior, this simplifies to:
    KG(x) = (f_best - mu) + 2 * sigma * phi((f_best - mu) / sigma)
    where phi is the standard normal PDF.
    """
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized distance
    z = (f_best - mu) / sigma_safe
    
    # Standard normal PDF
    pdf_z = norm.pdf(z)
    
    # Knowledge Gradient score
    # Higher score indicates better candidate
    kg_scores = (f_best - mu) + 2.0 * sigma_safe * pdf_z
    
    # Ensure finite output
    kg_scores = np.nan_to_num(kg_scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return kg_scores
# EVOLVE-BLOCK-END
