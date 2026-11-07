import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI)
    divided by cost^gamma.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Sanitize inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-9)
    
    # Calculate Expected Improvement (EI) for minimization
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Use stable calculation for EI
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # For large negative z, Phi(z) ~ 0, phi(z) ~ 0.
    # For large positive z, Phi(z) ~ 1, phi(z) ~ 0.
    
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    ei = (f_best - mu) * Phi + sigma * phi
    
    # Ensure EI is non-negative (numerical errors can cause tiny negatives)
    ei = np.maximum(ei, 0.0)
    
    # Cost penalty exponent
    # Inspiration 1 used EI / cost (gamma=1) and scored ~1.1
    # Current UCB / cost^1.5 scored ~0.56
    # Let's try gamma = 1.0 for EI
    gamma = 1.0
    
    cost_penalty = np.power(cost, gamma)
    
    score = ei / cost_penalty
    
    # Ensure output is finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
