import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement (PI)
    with explicit exploration and soft cost penalty.
    """
    t_start = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Sanitize inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    sigma = np.maximum(sigma, 1e-8)
    
    # Calculate Probability of Improvement (PI)
    # PI = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    pi = norm.cdf(z)
    
    # Calculate normalized uncertainty for exploration
    # This ensures the exploration term is scale-invariant
    sigma_max = np.max(sigma)
    if sigma_max > 1e-8:
        exploration = sigma / sigma_max
    else:
        exploration = np.zeros_like(sigma)
        
    # Combine PI and exploration
    # Beta = 1.0 balances exploitation (PI) and exploration (uncertainty)
    # This structural change replaces the magnitude-dependent EI with a bounded PI
    # plus an explicit uncertainty term, avoiding the numerical scaling issues of EI.
    beta = 1.0
    base_score = pi + beta * exploration
    
    # Apply soft cost penalty via exponentiation
    # cost is in [1, 2]. cost^-0.5 is in [1.0, 0.707].
    # This multiplicative penalty is smoother than division and less aggressive.
    cost_penalty = np.power(cost, -0.5)
    
    scores = base_score * cost_penalty
    
    # Deadline guard: return safe fallback if computation is unexpectedly slow
    if time.time() - t_start > 2.0:
        return exploration
        
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
