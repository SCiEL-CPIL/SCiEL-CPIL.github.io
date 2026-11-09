import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    # Time guard to ensure we finish well within the 300s limit
    start_time = time.time()
    
    # 1. Sanitize inputs to prevent NaN/Inf propagation
    # sigma must be positive for z calculation
    sigma = np.asarray(sigma, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to a small positive value to avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    
    # Replace any non-finite values in mu and sigma with safe defaults
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-8, posinf=1e6, neginf=1e-8)
    
    # Cost must be positive
    cost = np.maximum(cost, 1e-8)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1.0)
    
    # 2. Calculate Expected Improvement (EI) for minimization
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical underflow/overflow in cdf/pdf
    z = np.clip(z, -50, 50)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # 3. Add a small exploration bonus
    # This helps escape local optima where EI might be low due to low sigma
    # but the region is actually poor.
    exploration_bonus = 0.5 * sigma
    
    # 4. Combine and normalize by cost
    # Higher score is better.
    # Cost-aware: divide by cost.
    scores = (ei + exploration_bonus) / cost
    
    # 5. Ensure finite output
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Time check (should be instant, but good practice)
    if time.time() - start_time > 290.0:
        return scores
        
    return scores
