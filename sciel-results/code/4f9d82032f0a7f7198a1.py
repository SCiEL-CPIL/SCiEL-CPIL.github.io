import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive for calculation
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    # Using standard normal CDF and PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # UCB Exploration Term
    # Beta controls the trade-off between exploitation (EI) and exploration (UCB)
    # A value of 2.0 is a common heuristic for BO
    beta = 2.0
    
    # Add exploration bonus
    # This helps escape local minima and encourages sampling in uncertain regions
    hybrid_score = ei + beta * sigma
    
    # Cost-aware scaling
    # Divide by cost to penalize expensive evaluations
    # Cost is in [1, 2]
    scores = hybrid_score / cost
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
