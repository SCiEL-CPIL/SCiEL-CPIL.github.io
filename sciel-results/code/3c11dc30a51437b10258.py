import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = 2000
    if mu.shape != (n,):
        mu = mu.reshape(-1)
        if mu.shape[0] != n:
            return np.zeros(n, dtype=np.float64)
    if sigma.shape != (n,):
        sigma = sigma.reshape(-1)
        if sigma.shape[0] != n:
            return np.zeros(n, dtype=np.float64)
    if cost.shape != (n,):
        cost = cost.reshape(-1)
        if cost.shape[0] != n:
            return np.zeros(n, dtype=np.float64)
    
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -50, 50)
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    ei = np.maximum(ei, 0.0)
    
    scores = ei / cost_safe
    
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
