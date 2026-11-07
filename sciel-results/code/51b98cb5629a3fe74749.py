"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_start_time = time.time()
_DEADLINE_MARGIN = 5.0  # seconds before the 330s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    with a cheap-region exploration bonus.
    
    The key insight from the measured improvement (1.264): adding a small
    exploration term that favors cheaper regions prevents premature convergence
    while still allowing EI to drive exploitation toward the global minimum.
    """
    
    # Deadline guard: if we're close to the time limit, return a safe fallback
    elapsed = time.time() - _start_time
    if elapsed > (330.0 - _DEADLINE_MARGIN):
        # Fallback: return EI without any modifications to ensure finite output
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return np.nan_to_num(ei, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Compute Expected Improvement
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Standard EI calculation
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware modification:
    # cost is in [1, 2] where cost = 1 + mean(x_scaled)
    # We use a multiplicative cost penalty that is mild enough to not override
    # the EI signal but strong enough to slightly prefer cheaper regions.
    # 
    # Additionally, we add a small exploration bonus for cheap, uncertain regions.
    # This prevents the search from getting stuck in expensive local minima
    # while still allowing convergence to the global minimum.
    
    # Mild cost penalty: divide by (1 + alpha * (cost - 1))
    # alpha = 0.1 gives max penalty factor of 1.1 (10% reduction at highest cost)
    alpha_cost = 0.1
    cost_penalty = 1.0 + alpha_cost * (cost - 1.0)
    
    # Exploration bonus: favor cheap regions with high uncertainty
    # This term is small enough to not dominate EI but helps escape local optima
    # beta_explore = 0.05 * sigma * (1.0 / cost_penalty)
    # The (1/cost_penalty) makes the bonus stronger in cheaper regions
    
    # Combine: EI / cost_penalty + small exploration term
    exploration_term = 0.05 * sigma / cost_penalty
    
    scores = ei / cost_penalty + exploration_term
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
