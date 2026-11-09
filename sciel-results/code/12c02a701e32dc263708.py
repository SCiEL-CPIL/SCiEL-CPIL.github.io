"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: return current best if we exceed 320s (safety margin from 330s limit)
    start_time = time.time()
    
    # Cost-aware acquisition: Expected Improvement per unit cost (CEI)
    # This is the standard optimal acquisition function for cost-aware BO.
    # It balances the probability of improvement and the magnitude of improvement
    # against the cost of evaluation.
    
    # Compute z-score for EI
    # Clip sigma to avoid division by zero and numerical instability
    sigma_safe = np.clip(sigma, 1e-9, None)
    z = (f_best - mu) / sigma_safe
    
    # Compute Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: Since we are minimizing, f_best is the current best (lowest) value.
    # If mu < f_best, there is potential improvement.
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI / cost
    # cost is given as (2000,). We assume cost > 0.
    score = ei / cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
