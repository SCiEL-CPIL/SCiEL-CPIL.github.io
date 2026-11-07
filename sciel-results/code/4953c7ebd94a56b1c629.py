"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Calculates cost-penalized Expected Improvement (EI) for each candidate.
    
    Strategy:
    1. Use EI for minimization: EI = (f_best - mu) * Phi(z) + sigma * phi(z),
       where z = (f_best - mu) / sigma.
    2. Subtract a small cost penalty (lambda * cost) to guide the search toward
       cheaper regions without over-biasing.
    3. This approach balances exploitation (via EI) with cost awareness.
    """
    start_time = time.time()
    deadline = start_time + 325.0  # 5s safety margin
    
    # Ensure inputs are 1D arrays of float64 for performance and stability
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    cost = np.asarray(cost, dtype=np.float64).ravel()
    
    n = mu.shape[0]
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement for minimization
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Phi(z) = standard normal CDF
    # phi(z) = standard normal PDF
    from scipy.stats import norm
    Phi_z = norm.cdf(z)
    phi_z = norm.pdf(z)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Cost penalty: lambda * cost
    # Small lambda to avoid over-biasing toward cheap regions
    lambda_cost = 0.1
    cost_penalty = lambda_cost * cost
    
    # Final score: EI - cost_penalty
    score = ei - cost_penalty
    
    # Ensure finiteness: replace NaN/Inf with -inf so they are never selected
    score = np.where(np.isfinite(score), score, -np.inf)
    
    # Check deadline; if close, return immediately
    if time.time() > deadline - 1.0:
        return score
    
    return score
# EVOLVE-BLOCK-END
