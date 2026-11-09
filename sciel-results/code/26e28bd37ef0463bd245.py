import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Probability of Improvement (PI) per unit cost
    # PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    
    # Ensure cost is positive to avoid division issues
    cost_safe = np.maximum(cost, 1e-9)
    
    # Higher score is better
    score = pi / cost_safe
    
    return score
