# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Expected Improvement (EI) divided by cost.
    This approach uses the probabilistic expected improvement of the GP
    posterior, which is a distinct family from UCB. It balances exploration
    and exploitation by maximizing the expected reduction in the best observed
    value per unit of evaluation cost.
    """
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate the Z-score for Expected Improvement
    # Z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip Z to avoid numerical issues in CDF/PDF for extreme values
    z_clipped = np.clip(z, -10, 10)
    
    # Calculate CDF and PDF of standard normal distribution
    cdf = norm.cdf(z_clipped)
    pdf = norm.pdf(z_clipped)
    
    # Expected Improvement formula for minimization:
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI per unit cost
    # Higher score is better
    score = ei / cost
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
