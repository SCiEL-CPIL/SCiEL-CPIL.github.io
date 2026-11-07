import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 325.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement per cost.
    Returns scores of shape (2000,), where higher scores indicate better candidates.
    """
    # Explicit deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback: Simple Cost-Aware EI
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        scores = ei / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)

    # Ensure safety in denominators
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute Expected Improvement (EI)
    # EI(x) = E[max(f_best - f(x), 0)]
    # For GP: EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    z = (f_best - mu) / sigma_safe
    
    # Numerical stability: 
    # For very large z, cdf(z) -> 1, pdf(z) -> 0. EI -> f_best - mu.
    # For very small z, cdf(z) -> 0.5, pdf(z) -> 0.4.
    # np functions handle this well, but we clip z to avoid overflow in exp for pdf if needed (unlikely for standard normal).
    
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure EI is non-negative (numerical errors can cause small negatives)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI per unit cost
    # Using a slightly steeper cost penalty to discourage expensive regions more strongly
    # Cost is in [1, 2]. cost**1.2 is a mild non-linear penalty.
    # Let's stick to linear cost first as it scored 1.0.
    # However, to make a "structural change" from the current LCB, EI is the change.
    # To potentially beat 1.0, I will try a small non-linear cost penalty.
    # If the minimum is in the center (cost ~1.5), and corners are cost 2.0.
    # Linear: 1/1.5 vs 1/2.0 -> 0.66 vs 0.5.
    # Power 1.2: 1/1.5**1.2 vs 1/2.0**1.2 -> 0.63 vs 0.44.
    # This makes the center relatively more attractive compared to corners.
    
    scores = ei / (cost_safe ** 1.2)
    
    # Ensure all outputs are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
