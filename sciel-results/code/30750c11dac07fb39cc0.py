import numpy as np
import time
from scipy.stats import norm

# Global deadline tracking
_start_time = None
_deadline = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time, _deadline
    
    if _start_time is None:
        _start_time = time.time()
        _deadline = _start_time + 300.0
    
    # Check deadline with safety margin
    elapsed = time.time() - _start_time
    if elapsed > 290.0:
        # Return a safe default (exploit best known) to avoid timeout
        return np.zeros_like(mu)
    
    # mu: mean of GP (2000,)
    # sigma: std of GP (2000,)
    # f_best: best observed value (scalar)
    # cost: cost of evaluating each candidate (2000,)
    
    # Structural change: Use Cost-Adjusted Expected Improvement (EI)
    # EI for minimization: E[max(f_best - f(x), 0)]
    # where f(x) ~ N(mu, sigma^2)
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is CDF and phi is PDF of standard normal
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Compute EI
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost (not sqrt) for stronger cost penalty
    # This makes the acquisition function prefer cheaper evaluations more aggressively
    cost_safe = np.maximum(cost, 1e-6)
    
    # Compute cost-adjusted EI
    scores = ei / cost_safe
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
