import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Expected Improvement (EI) for minimization
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is CDF and phi is PDF of standard normal.
    
    # Cost-aware: Divide by cost to prefer cheaper evaluations.
    # This is the standard "Cost-Weighted EI" or "EI per unit cost".
    
    # The parent program (score 1.0) likely used this or something very similar.
    # The current program (score 2.74) uses PI / cost, which is worse.
    # Let's revert to EI / cost, which is a strong baseline.
    
    # Handle sigma = 0 case to avoid division by zero.
    # If sigma is 0, EI is 0 if mu >= f_best, else f_best - mu.
    # But GP sigma is rarely exactly 0.
    
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0)
    
    # Cost-aware score
    score = ei / cost
    
    return score
