import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    # Apply cost penalty only to exploration term to preserve exploitation signal
    
    # Ensure sigma is positive to avoid division by zero or invalid z
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Standard normal CDF and PDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Cost-aware scaling: penalize exploration (uncertainty) more than exploitation
    # The exploitation term (f_best - mu) stays strong; exploration (sigma * phi) gets damped by cost
    # Use a moderate cost penalty that scales with cost relative to minimum cost (1.0)
    cost_ratio = cost / 1.0  # cost ranges from 1.0 to 2.0
    
    # Apply cost penalty to the exploration component
    # We can decompose EI into exploitation and exploration parts:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Penalize the second term (exploration) by 1/cost_ratio
    exploration_term = sigma_safe * phi_z
    exploitation_term = (f_best - mu) * Phi_z
    
    # Dampen exploration based on cost
    cost_damp = 1.0 / (1.0 + 0.3 * (cost_ratio - 1.0))
    
    scores = exploitation_term + cost_damp * exploration_term
    
    # Handle any potential non-finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
