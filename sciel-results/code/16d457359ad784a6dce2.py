"""Rewritten program using a cost-aware Expected Improvement (EI) acquisition."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function.
    
    For minimization, EI measures the expected amount by which the function value
    at x exceeds the current best f_best (i.e., f_best - f(x)).
    
    EI(x) = (f_best - mu) * Phi(u) + sigma * phi(u)
    where u = (f_best - mu) / sigma
    
    We divide by cost to get cost-aware EI (cEI).
    Higher scores indicate better candidates to evaluate.
    """
    # Ensure sigma is strictly positive to avoid division by zero
    # A very small epsilon is used to prevent division by zero, but in practice
    # sigma from a GP is rarely exactly 0 unless at observed points.
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate the standardized improvement u
    u = (f_best - mu) / sigma_safe
    
    # Calculate the CDF and PDF of the standard normal distribution
    cdf_u = norm.cdf(u)
    pdf_u = norm.pdf(u)
    
    # Calculate Expected Improvement
    # Note: If mu > f_best, u is negative. 
    # (f_best - mu) is negative. Phi(u) is small. sigma*phi(u) is positive.
    # The sum should be non-negative.
    ei_vals = (f_best - mu) * cdf_u + sigma_safe * pdf_u
    
    # Ensure EI is non-negative (numerical stability)
    ei_vals = np.maximum(ei_vals, 0.0)
    
    # Cost-aware score: EI per unit cost
    scores = ei_vals / cost
    
    # Ensure all outputs are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
