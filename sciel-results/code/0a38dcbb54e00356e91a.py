import numpy as np
import time
from scipy.stats import norm

_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: 300s limit
    if time.time() - _start_time > 295:
        return np.zeros(2000)

    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement for minimization
    # Improvement I(x) = max(0, f_best - f(x))
    # E[I(x)] = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues in CDF/PDF
    z = np.clip(z, -10, 10)
    
    Phi_z = norm.cdf(z)
    phi_z = norm.pdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Multiplicative Cost Gate
    # score = EI * exp(-k * cost)
    
    # Calibration of k:
    # Cost c(x) = 1 + mean(x_scaled). Range [1, 2].
    # We want to penalize high cost.
    # If k is too small, cost is ignored.
    # If k is too large, only lowest cost points are considered, ignoring EI.
    # 
    # Typical EI scale:
    # Early: f_best ~ 0.5, mu ~ 0, sigma ~ 0.5. z ~ 1. EI ~ 0.5*0.84 + 0.5*0.24 ~ 0.5.
    # Late: f_best ~ -0.7, mu ~ -0.8, sigma ~ 0.1. z ~ 10. EI ~ 0.1.
    # 
    # Cost range [1, 2].
    # exp(-k*1) vs exp(-k*2). Ratio is exp(-k).
    # If k=1, ratio is 0.37. A point with cost 2 gets 37% of the score of a point with cost 1 (for same EI).
    # If k=2, ratio is 0.13.
    # If k=0.5, ratio is 0.61.
    # 
    # The minimum is in the inner 80% (x_scaled in [0.1, 0.9]).
    # Cost of minimum is between 1.1 and 1.9.
    # We want to explore the inner region but not be strictly forced to the corner (0,0) which has cost 1.
    # However, the corners are not where the minimum is.
    # A moderate penalty is good.
    # Let's try k = 1.0.
    
    k = 1.0
    gate = np.exp(-k * cost)
    
    score = ei * gate
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
