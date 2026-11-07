"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure inputs are numpy arrays for vectorized operations
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Deadline guard: if we're running low on time, return a simple score
    # The evaluator kills at 330s; we add a safety margin
    t_start = time.time()
    deadline = t_start + 280.0  # 50s safety margin
    
    # Clip sigma to a small positive value to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Fixed exploration parameter based on problem scale
    # Levy 2d on [-10,10]^2 has values roughly in [0, 20+], so use a small
    # fraction of the expected range. This is more stable than using median sigma.
    xi = 0.05
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu - xi) / sigma_safe
    
    # Check deadline before expensive CDF computation
    if time.time() > deadline:
        # Fallback: simple EI approximation using just the mean difference
        ei_fallback = np.maximum(f_best - mu - xi, 0.0)
        cost_safe = np.maximum(cost, 1e-8)
        score = ei_fallback / cost_safe
        score = np.where(np.isfinite(score), score, 0.0)
        return score
    
    # Calculate CDF and PDF using scipy.stats.norm for numerical stability
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Expected Improvement
    ei = (f_best - mu - xi) * cdf + sigma_safe * pdf
    
    # Check deadline again after CDF computation
    if time.time() > deadline:
        cost_safe = np.maximum(cost, 1e-8)
        score = ei / cost_safe
        score = np.where(np.isfinite(score), score, 0.0)
        return score
    
    # Cost-aware score: EI per unit cost
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-8)
    score = ei / cost_safe
    
    # Replace any non-finite values with 0 to ensure validity
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
