import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Calculate Expected Improvement
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    sigma_safe = np.where(sigma < 1e-9, 1e-9, sigma)
    z = (f_best - mu) / sigma_safe
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Handle cases where sigma was very small (EI should be max(0, f_best - mu))
    # norm.cdf(z) for large z is 1, norm.pdf(z) is 0.
    # If sigma -> 0, z -> inf (if f_best > mu), EI -> f_best - mu.
    # If sigma -> 0, z -> -inf (if f_best < mu), EI -> 0.
    # The formula above is stable enough with sigma_safe.
    
    # Quadratic cost penalty
    # Cost is in [1, 2]
    # Penalty = lambda * cost^2
    lambda_cost = 1.0
    penalty = lambda_cost * (cost ** 2)
    
    # Small uncertainty bonus to encourage exploration if EI is flat
    beta_explore = 0.1
    bonus = beta_explore * sigma_safe
    
    score = ei + bonus - penalty
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
