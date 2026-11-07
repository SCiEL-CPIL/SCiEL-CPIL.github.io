import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma
    
    # CDF and PDF of the standard normal distribution
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * P(Y < f_best) + sigma * E[max(f_best - Y, 0) | Y ~ N(mu, sigma^2)]
    ei = (f_best - mu) * cdf + sigma * pdf
    
    # Add an exploration bonus proportional to uncertainty
    # This helps escape local minima and explore the 6D landscape more effectively
    exploration_weight = 0.1
    score = ei + exploration_weight * sigma
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
