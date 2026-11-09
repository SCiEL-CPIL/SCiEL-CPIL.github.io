import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure inputs are arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Small epsilon for numerical stability
    eps = 1e-9
    
    # Calculate z-score
    # z = (f_best - mu) / sigma
    # We want to handle sigma ~ 0 carefully
    z = (f_best - mu) / (sigma + eps)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Ensure EI is non-negative (numerical errors can cause tiny negative values)
    ei = np.maximum(ei, 0.0)
    
    # Exploration term (UCB-like)
    # The knowledge suggests blending with a UCB term with kappa=1.5.
    # In minimization, the exploration component of the confidence bound is kappa * sigma.
    kappa = 1.5
    exploration = kappa * sigma
    
    # Combine EI and exploration
    # We want to maximize this score
    score = ei + exploration
    
    # Cost-aware: divide by cost
    # cost is shape (2000,), score is shape (2000,)
    # Avoid division by zero, though cost >= 1
    score = score / (cost + eps)
    
    # Ensure finite scores
    # Replace any non-finite values with 0 or a very small number
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
