"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    n = len(demands)
    d = np.asarray(demands, dtype=np.float64)
    pair_sum = d[:, None] + d[None, :]
    waste = capacity - pair_sum
    
    # Start with base: pairs that fit get a value based on how tight they are
    # Use inverse of (waste + 1) to reward tight fits
    base = np.where(waste >= 0, 1.0 / (waste + 1.0), 1e-9)
    
    # Apply power to sharpen preference for tight fits
    power = 8.0
    base = np.power(base, power)
    
    # Add explicit bonus for very tight fits (waste < 5)
    tight_bonus = np.where((waste >= 0) & (waste < 5), 2.0 / (waste + 1.0), 0.0)
    tight_bonus = np.power(tight_bonus, power)
    
    # Combine
    result = base + tight_bonus
    
    # Ensure non-finite values are replaced
    result = np.where(np.isfinite(result), result, 1e-9)
    
    # Ensure all non-positive become 1e-9
    result = np.where(result > 0, result, 1e-9)
    
    return result
# EVOLVE-BLOCK-END
