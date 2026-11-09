"""Cost-aware acquisition using Expected Improvement (EI) normalized by cost."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 320s wall-clock safety margin (limit is 330s)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for Bayesian optimization.
    
    mu, sigma, f_best, cost: each shape (2000,)
    f_best: scalar (lowest observed value)
    cost: per-candidate cost = 1 + mean(x_scaled)
    
    Returns: 2000 finite scores (higher = evaluate next)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against zero/negative values
    cost_safe = np.maximum(cost, 1e-9)
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    diff = f_best - mu
    z = diff / sigma_safe
    
    # Calculate EI components
    phi_z = norm.pdf(z)
    phi_z = np.where(np.isfinite(phi_z), phi_z, 0.0)
    
    Phi_z = norm.cdf(z)
    Phi_z = np.where(np.isfinite(Phi_z), Phi_z, 0.0)
    
    ei = diff * Phi_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost to get expected improvement per unit cost
    score = ei / cost_safe
    
    # Ensure all scores are finite and non-negative
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
