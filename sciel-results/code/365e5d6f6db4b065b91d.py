import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # z = (f_best - mu) / sigma
    
    # Handle zero sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Compute EI components
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Add a small uncertainty bonus to encourage exploration in high-sigma regions
    # This helps escape local optima in the multimodal Shekel 4d function
    # The bonus is scaled to be small relative to typical EI values
    sigma_max = np.max(sigma) if np.max(sigma) > 1e-12 else 1.0
    uncertainty_bonus = 0.1 * (sigma / sigma_max) * np.max(ei) if np.max(ei) > 1e-12 else 0.0
    
    blended = ei + uncertainty_bonus
    
    # Cost-aware score
    # Cost is in [1, 2]. Use a moderate penalty to balance cost vs. improvement
    # Linear penalty is less aggressive than quadratic, allowing more exploration
    cost_safe = np.maximum(cost, 1.0)
    scores = blended / cost_safe
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    if time.time() - start_time > 320:
        return np.zeros_like(mu)
        
    return scores
