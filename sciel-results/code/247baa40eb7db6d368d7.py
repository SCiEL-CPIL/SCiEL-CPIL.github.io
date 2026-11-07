import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure inputs are arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Safe sigma
    sigma_safe = sigma + 1e-9
    
    # Z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    # For minimization: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware weighting
    # Using a power of 0.7 to soften the cost penalty compared to baseline (power 1.0)
    alpha = 0.7
    cost_weight = cost ** alpha
    
    # Exploration term (UCB-like)
    # Adding a multiple of sigma encourages exploration in uncertain regions
    # This helps escape local minima and cover the space more effectively
    exploration_lambda = 1.0
    score = (ei + exploration_lambda * sigma) / cost_weight
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
