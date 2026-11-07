import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI)
    with a linear cost penalty.
    
    Strategy:
    1. Compute Expected Improvement (EI): E[max(f_best - f(x), 0)].
       This rewards points with high potential for significant improvement,
       unlike PI which only rewards the probability of any improvement.
    2. Apply a linear cost penalty: subtract beta * cost.
       This directly models the cost of evaluation as a reduction in net benefit.
       The cost is in [1, 2], so beta controls the trade-off.
    3. Return the net score: EI - beta * cost.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)
    
    # Safety: ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Calculate z-score for improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF.
    
    # Clip z to avoid numerical issues, though scipy is generally robust
    z_clipped = np.clip(z, -50, 50)
    
    phi_z = norm.pdf(z_clipped)
    Phi_z = norm.cdf(z_clipped)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Linear cost penalty
    # Cost is in [1, 2]. 
    # If f_best is around 0.1 (typical for Griewank near min), EI might be ~0.01-0.05.
    # We need beta such that beta * cost is comparable to EI.
    # If beta = 0.1, cost=1 -> 0.1, cost=2 -> 0.2.
    # This is quite strong if EI is small.
    # Let's try a smaller beta to allow exploration, but still prefer cheaper points.
    # beta = 0.05 seems reasonable.
    beta = 0.05
    
    cost_penalty = beta * cost
    
    # Net Score
    score = ei - cost_penalty
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # If all scores are identical (e.g., very early on or flat GP), add small noise
    if np.max(score) - np.min(score) < 1e-15:
        score = score + np.random.randn(len(score)) * 1e-10
        
    return score
