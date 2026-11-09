import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Cost-Aware Probability of Improvement (PI) for minimization
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Z-score
    Z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    PI = norm.cdf(Z)
    
    # Ensure PI is in [0, 1]
    PI = np.clip(PI, 0.0, 1.0)
    
    # Cost-aware score: PI / cost
    score = PI / cost
    
    # Ensure finite scores
    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
