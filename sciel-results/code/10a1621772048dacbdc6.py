"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: keep a module-level start time to enforce the 120s limit.
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # 5s safety margin before 120s kill

# Weibull shape parameter
_WEIBULL_K = 2.0
# Base scale parameter (adapted per-item below)
_WEIBULL_LAMBDA_BASE = 12.0
# Consolidation penalty weight: stronger than before to prevent Worst-Fit drift
_BETA = 0.35

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Returns an array of scores (higher is better for selection).
    
    Strategy: Adaptive distribution-aware "Expected Dead Space" minimization.
    
    DeadSpace(r) = r * exp(-(r/lambda)^k)
    with lambda adapted to the current item size: larger items imply a larger
    effective scale, shifting the dead-zone peak.
    
    Score = -(dead_space + beta * r)
    The beta*r term ensures a single peak and prevents pure Worst-Fit.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        remaining = bins - item
        mask = remaining >= 0
        scores = np.full(len(bins), -np.inf)
        if np.any(mask):
            scores[mask] = -remaining[mask]
        return scores

    # Calculate remaining capacity
    remaining = bins - item
    
    # Identify valid bins (enough capacity)
    valid = remaining >= 0
    
    # Initialize scores with -inf for invalid bins
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(valid):
        return scores
        
    valid_rems = remaining[valid]
    
    # Clip r to non-negative
    r = np.maximum(valid_rems, 0.0)
    
    # Adapt the Weibull scale: larger items suggest a broader distribution
    # of future items, so the "dead zone" shifts to larger remainders.
    # Use a smooth adaptation: lambda = base * (1 + item / capacity_scale)
    # capacity_scale ~ 100 (bin capacity), so for item=50, lambda ~ base*1.5
    lambda_adapted = _WEIBULL_LAMBDA_BASE * (1.0 + item / 100.0)
    
    # Calculate survival probability S(r) = exp(-(r/lambda)^k)
    ratio = r / lambda_adapted
    power_term = np.power(ratio, _WEIBULL_K)
    survival = np.exp(-power_term)
    
    # Dead space metric
    dead_space = r * survival
    
    # Combined cost: dead space + consolidation penalty
    combined_cost = dead_space + _BETA * r
    
    scores[valid] = -combined_cost
    
    return scores
# EVOLVE-BLOCK-END
