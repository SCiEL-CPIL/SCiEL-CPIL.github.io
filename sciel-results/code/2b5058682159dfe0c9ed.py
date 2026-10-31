"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global variables to track item statistics for adaptive heuristic
_item_count = 0
_item_sum = 0.0
_start_time = time.time()

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    global _item_count, _item_sum, _start_time
    
    # Time guard: if we are close to the limit, fall back to a simple, fast heuristic
    # to ensure we return something. 150s limit, keep 5s margin.
    if time.time() - _start_time > 145.0:
        # Fallback: Best Fit (minimize remainder)
        scores = np.full(len(bins), -np.inf)
        valid = bins >= item
        if np.any(valid):
            scores[valid] = -(bins[valid] - item)
        return scores

    # Update statistics
    _item_count += 1
    _item_sum += item
    
    # Estimate mean
    mean = _item_sum / _item_count if _item_count > 0 else 10.0
    
    # Weibull parameters
    # Assume shape k=1.5 (typical for sizes)
    # Scale lambda = mean / Gamma(1 + 1/k)
    # For k=1.5, Gamma(1.666) ~ 0.903
    # lambda = mean / 0.903
    k = 1.5
    gamma_val = 0.903  # Approx Gamma(1 + 1/1.5)
    lambda_ = mean / gamma_val
    
    scores = np.full(len(bins), -np.inf)
    valid = bins >= item
    if np.any(valid):
        rems = bins[valid] - item
        
        # Calculate Weibull CDF: F(r) = 1 - exp(-(r/lambda)^k)
        # We want to balance minimizing remainder (Best Fit) with maximizing fillability (First Fit-like)
        # Score = -rem + beta * F(rem)
        # If beta is large, we prefer larger rems (First Fit)
        # If beta is 0, we prefer smaller rems (Best Fit)
        
        # Let's set beta proportional to the mean item size to make it scale-invariant
        # A reasonable beta might be 0.5 * mean
        beta = 0.5 * mean
        
        if lambda_ > 0:
            # Clip rems to avoid numerical issues, though rems >= 0
            # F(rem)
            z = np.power(rems / lambda_, k)
            # For large z, exp(-z) is 0, F is 1.
            # For small z, exp(-z) ~ 1-z, F ~ z.
            cdf_vals = 1.0 - np.exp(-z)
            
            scores[valid] = -rems + beta * cdf_vals
        else:
            # Fallback if mean is 0
            scores[valid] = -rems
            
    return scores
# EVOLVE-BLOCK-END
