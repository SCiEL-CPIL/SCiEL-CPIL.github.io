"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement per unit cost.
    
    This approach directly optimizes the marginal return on each unit of budget spent,
    naturally balancing high-improvement regions with cheaper evaluation costs.
    """
    t0 = time.time()
    
    # Safety check for inputs
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)) or not np.isfinite(f_best):
        return np.zeros(len(mu))
        
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1.0)
    
    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)  # EI should be non-negative
    
    # Cost-aware acquisition: EI per unit cost
    # This directly optimizes the expected improvement per unit of budget spent
    score = ei / cost_safe
    
    # Add a small exploration bonus based on uncertainty, weighted by cost
    # This helps escape local optima while still being cost-aware
    # The exploration term is sigma / cost, which favors uncertain but cheap points
    exploration_weight = 0.1
    exploration = exploration_weight * sigma_safe / cost_safe
    score = score + exploration
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline guard
    if time.time() - t0 > 280:
        # Fallback: Simple EI / cost (less aggressive but safe)
        fallback = ei / cost_safe
        fallback = np.where(np.isfinite(fallback), fallback, 0.0)
        return fallback
        
    return score
# EVOLVE-BLOCK-END
