"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    n = len(demands)
    d = demands.astype(np.float64)
    
    # Pair sums
    pair = d[:, None] + d[None, :]
    
    # Fill ratio for pairs that fit
    fill_ratio = np.where(pair <= capacity, pair / capacity, 0.0)
    
    # Reward complementary fits: pairs that fill the bin well
    # Use a power to amplify high fill ratios (non-linear relationship)
    # High power makes the heuristic strongly prefer near-complete fills
    power = 8.0
    heuristic = np.where(pair <= capacity, fill_ratio ** power, 0.0)
    
    # Add a small bonus for medium-sized items (they're more flexible)
    # Items around capacity/2 are most versatile
    mid_bonus = 1.0 - np.abs(d - capacity / 2.0) / (capacity / 2.0)
    mid_bonus = np.clip(mid_bonus, 0.1, 1.0)
    
    # Combine: fill ratio power dominates, with slight flexibility bonus
    heuristic = heuristic * (1.0 + 0.3 * (mid_bonus[:, None] + mid_bonus[None, :]) / 2.0)
    
    # Ensure non-fitting pairs get minimal value
    heuristic = np.where(pair <= capacity, heuristic, 1e-9)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
