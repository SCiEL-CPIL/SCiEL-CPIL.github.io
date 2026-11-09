import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-8)
    cost = np.maximum(np.asarray(cost, dtype=np.float64), 1e-8)
    
    # Expected Improvement for minimization
    improvement = f_best - mu
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure ei is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI per unit cost
    score = ei / cost
    
    # Small exploration bonus to avoid premature convergence
    # Scaled to be negligible when EI is large
    exploration = 0.005 * sigma / cost
    
    return score + exploration
