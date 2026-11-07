import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Use Expected Improvement (EI) for minimization.
    # EI measures the expected amount of improvement over the current best f_best.
    # For minimization: EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma.
    # We divide by cost to account for evaluation costs.
    
    # Avoid division by zero in z calculation
    eps = 1e-8
    sigma_safe = np.where(sigma > eps, sigma, eps)
    z = (f_best - mu) / sigma_safe
    
    # Calculate standard normal PDF and CDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Calculate Expected Improvement
    # Note: If sigma is very small, the formula approximates max(f_best - mu, 0)
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Cost-aware score: Higher is better
    # We want high EI and low cost
    scores = ei / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
