# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Approach: Cost-Weighted Expected Improvement (EI) for minimization.
    # EI quantifies the expected improvement over the current best value f_best.
    # We divide by cost to prefer cheaper evaluations that offer similar expected gains.
    
    # Ensure sigma is positive to avoid division by zero in Z calculation
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Calculate Z score: (f_best - mu) / sigma
    # If f_best > mu, Z is positive (we expect improvement)
    # If f_best < mu, Z is negative (we expect no improvement)
    Z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula for minimization:
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # Where Phi is the CDF and phi is the PDF of the standard normal distribution.
    
    # Use scipy.stats.norm for numerical stability
    phi_Z = norm.pdf(Z)
    Phi_Z = norm.cdf(Z)
    
    EI = (f_best - mu) * Phi_Z + sigma_safe * phi_Z
    
    # Clip to 0 to handle floating point errors that might yield tiny negative values
    EI = np.maximum(EI, 0.0)
    
    # Normalize by cost to get improvement per unit cost
    score = EI / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
