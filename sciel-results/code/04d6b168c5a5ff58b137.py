import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Expected Improvement (EI) for minimization
    # EI = E[max(f_best - f(x), 0)]
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Z-score
    z = (f_best - mu) / sigma_safe
    
    # Standard Normal CDF and PDF
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Expected Improvement
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Cost-aware score: higher EI per unit cost is better
    score = ei / cost_safe
    
    return score
