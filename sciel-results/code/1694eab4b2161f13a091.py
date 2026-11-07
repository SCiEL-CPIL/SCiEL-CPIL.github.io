"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Expected Improvement per unit cost with exploration bonus.
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Deadline guard: return simple EI/cost if we're running out of time
    elapsed = time.time() - _START_TIME
    if elapsed > 280:
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        z = np.clip(z, -15, 15)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        cost_safe = np.maximum(cost, 1e-8)
        result = ei / cost_safe
        return np.where(np.isfinite(result), result, 0.0)
    
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -15, 15)
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware EI
    cost_safe = np.maximum(cost, 1e-8)
    ei_per_cost = ei / cost_safe
    
    # Exploration bonus based on uncertainty
    # Scale exploration by how much we've explored (use f_best as proxy)
    # Early on: more exploration; later: more exploitation
    # f_best starts high (worst case ~0) and decreases toward -0.8
    # Use a smooth transition based on improvement
    improvement = f_best  # negative means we've found good values
    # beta: higher when f_best is worse (less explored)
    beta = 0.3 + 0.7 * np.clip(-improvement / 1.0, 0, 1)
    
    # Uncertainty bonus
    exploration = beta * sigma / cost_safe
    
    # Combine: EI per cost + exploration bonus
    combined = ei_per_cost + exploration
    
    # Ensure finite values
    combined = np.where(np.isfinite(combined), combined, 0.0)
    
    return combined
# EVOLVE-BLOCK-END
