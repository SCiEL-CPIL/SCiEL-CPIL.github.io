import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is non-negative and handle very small values
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Add a small exploration bonus to encourage exploration
    # This helps if the GP is overconfident or stuck
    exploration_bonus = 0.1 * sigma
    
    # Cost-aware score
    # Higher is better
    # Divide by cost to penalize expensive points
    # Ensure cost is not zero (min cost is 1)
    score = (ei + exploration_bonus) / cost
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)
    
    return score
