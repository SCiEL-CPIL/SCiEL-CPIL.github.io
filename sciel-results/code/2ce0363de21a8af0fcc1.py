import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Cost-Adjusted Expected Improvement (CAEI)
    # Standard EI for minimization: E[max(f_best - f(x), 0)]
    
    # Avoid division by zero in Z
    sigma_safe = np.where(sigma > 1e-8, sigma, 1e-8)
    
    Z = (f_best - mu) / sigma_safe
    
    # Standard Normal CDF and PDF
    Phi_Z = norm.cdf(Z)
    phi_Z = norm.pdf(Z)
    
    # Expected Improvement
    EI = (f_best - mu) * Phi_Z + sigma_safe * phi_Z
    
    # Ensure EI is non-negative (numerical stability)
    EI = np.maximum(EI, 0.0)
    
    # Cost adjustment
    # Using sqrt(cost) to balance exploration and cost
    # Alternatively, just EI / cost if we want strict cost-efficiency
    # Let's try EI / sqrt(cost) first, as it's robust
    score = EI / np.sqrt(cost)
    
    return score
