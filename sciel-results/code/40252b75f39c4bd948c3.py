import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using PI + Variance term.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety checks
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Z-score
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid extreme values in PDF/CDF
    z = np.clip(z, -10, 10)
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Variance term (from EI)
    # This term is high when z is near 0 (uncertain about improvement)
    var_term = sigma_safe * norm.pdf(z)
    
    # Weight for the variance term.
    # If lambda is 0, it's pure PI.
    # If lambda is 1, it's PI + VarTerm.
    # We want to stay close to PI (score 3.17) but add a bit of exploration.
    # Let's try lambda = 0.5 initially.
    lambda_var = 0.5
    
    score = (pi + lambda_var * var_term) / cost_safe
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
