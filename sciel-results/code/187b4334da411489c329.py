"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state to track elapsed time and call count
_T0 = [None]
_CALL_COUNT = [0]

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a blend of EI and pure exploration (sigma),
    with an adaptive decay rate for the exploration component.
    
    Args:
        mu: Array of GP means (2000,)
        sigma: Array of GP standard deviations (2000,)
        f_best: Best observed function value (scalar)
        cost: Array of evaluation costs (2000,)
        
    Returns:
        Array of scores (2000,), higher is better.
    """
    # Initialize timer on first call
    if _T0[0] is None:
        _T0[0] = time.time()
    
    elapsed = time.time() - _T0[0]
    _CALL_COUNT[0] += 1
    
    # Deadline guard with safety margin
    if elapsed > 315.0:
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        # Fallback: simple exploration-based score
        score = sigma_safe / cost_safe
        return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Sanitize inputs
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate Expected Improvement (EI) for minimization
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Adaptive Exploration Weight
    # We want strong exploration early on, decaying smoothly.
    # Previous used linear decay 0.3 * (1 - t). 
    # New: Exponential decay starting higher (1.0) and decaying with a time constant.
    # Total budget ~30 cost. Initial ~7.5. ~22.5 remaining.
    # Assume ~15-20 evaluations.
    # Let's decay over the expected number of evaluations or time.
    # Using time is safer.
    # Time constant: ~50 seconds? So at 50s, weight is 1/e ~ 0.37. At 100s, 0.13.
    # This maintains exploration longer than the previous linear 0.3 drop.
    time_constant = 60.0 
    exploration_weight = 1.0 * np.exp(-elapsed / time_constant)
    
    # Combine EI and Exploration
    # Score = (EI + weight * sigma) / cost
    # This is equivalent to Cost-Aware EI + Cost-Aware UCB-like term
    numerator = ei + exploration_weight * sigma_safe
    score = numerator / cost_safe
    
    # Numerical safety
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
