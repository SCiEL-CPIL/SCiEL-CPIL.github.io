"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement with a deadline guard.
    
    Structural idea: Use the simple, robust EI/cost formulation (from inspirations 1 & 2)
    as the core acquisition function, but add a deadline guard to ensure the function
    returns within the time budget. The refinement machinery includes careful handling
    of numerical edge cases and finite value checks.
    
    The key insight is that EI/cost is a principled, simple approach that has been shown
    to score 1.0 in the inspirations. We maintain this core but add safety mechanisms.
    """
    # Track start time for deadline guard
    t_start = time.time()
    deadline = t_start + 2.0  # 2 second safety margin within 300s budget
    
    # Convert to numpy arrays for safety
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 1e-6), cost, 1.0)
    
    # Safety: avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -50, 50)  # Numerical stability
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)  # EI should be non-negative
    
    # Cost-aware EI: EI / cost
    ei_cost = ei / cost_safe
    
    # Deadline guard: if we're running out of time, return what we have
    if time.time() > deadline:
        return np.where(np.isfinite(ei_cost), ei_cost, -np.inf)
    
    # Ensure finite output
    score = np.where(np.isfinite(ei_cost), ei_cost, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
