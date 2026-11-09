import numpy as np
from scipy.stats import norm
import time

_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: 300s limit
    # The function is called once per iteration. 
    # If we are close to the limit, we should return safe values.
    # However, the computation is vectorized and extremely fast (<1ms).
    # This guard is primarily for safety in case of unexpected overhead.
    elapsed = time.time() - _start_time
    if elapsed > 295:
        return np.zeros_like(mu)

    sigma_safe = np.maximum(sigma, 1e-8)
    
    # z-score for EI
    # For minimization, EI = E[max(f_best - f(x), 0)]
    # f(x) ~ N(mu, sigma^2)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Standard EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Strategy: Multiplicative exponential cost gate
    # Score = EI * exp(-k * cost)
    # Cost range is [1, 2].
    # We want to suppress high cost (2) relative to low cost (1).
    # If k is too small, cost is ignored. If k is too large, only cost=1 points are considered.
    # Let's calibrate k such that a point with cost=2 is discounted by ~exp(-k).
    # A reasonable discount for "expensive" points might be 0.5 or 0.25.
    # exp(-k*2) / exp(-k*1) = exp(-k).
    # If we want a factor of 0.5 difference between cost 1 and 2, exp(-k) = 0.5 => k = ln(2) ≈ 0.69.
    # If we want a factor of 0.25, exp(-k) = 0.25 => k = ln(4) ≈ 1.38.
    # Let's try k = 1.0 as a moderate suppression.
    # This makes cost=2 points worth ~37% of their EI compared to cost=1 points (exp(-1) ≈ 0.37).
    
    k = 1.0
    cost_gate = np.exp(-k * cost)
    
    score = ei * cost_gate
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
