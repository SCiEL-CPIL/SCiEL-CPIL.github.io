"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid of Normalized Expected Improvement (NEI)
    and Probability of Improvement (POI).
    
    NEI = z * Phi(z) + phi(z)
    POI = Phi(z)
    Score = (NEI + POI) / cost
    
    This approach is robust to the scale of the objective function and balances
    exploitation (NEI) with exploration (POI/PDF components).
    """
    eps = 1e-9
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Clip z to a reasonable range to prevent numerical issues in exp/cdf
    # norm.cdf and norm.pdf handle large values, but clipping ensures stability
    # and prevents potential overflow in intermediate calculations if any
    z_clipped = np.clip(z, -10, 10)
    
    # Compute CDF and PDF
    cdf = norm.cdf(z_clipped)
    pdf = norm.pdf(z_clipped)
    
    # Normalized Expected Improvement (NEI)
    # This is the EI scaled by sigma, removing dependence on the absolute scale of mu/f_best
    nei = z_clipped * cdf + pdf
    
    # Probability of Improvement (POI)
    poi = cdf
    
    # Hybrid numerator
    # Weights: 1.0 for NEI, 1.0 for POI
    # This encourages both high expected improvement and high probability of improvement
    score_num = nei + poi
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, eps)
    
    # Cost-aware score
    score = score_num / cost_safe
    
    # Handle any potential NaNs or Infs
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
