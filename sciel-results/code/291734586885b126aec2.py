import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-12)
    
    # Expected Improvement for minimization
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues in norm.cdf/pdf
    z_clipped = np.clip(z, -10, 10)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Cost-aware: EI per unit cost
    score = ei / cost_safe
    
    # Add small exploration bonus to encourage diversity
    # This helps escape local optima in the acquisition landscape
    exploration = 0.01 * sigma_safe / cost_safe
    
    score = score + exploration
    
    # Ensure finite outputs
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
