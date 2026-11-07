"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    """
    Compute an item-by-item attractiveness matrix for placing item j right after item i.
    
    The heuristic rewards pairs that fill bins efficiently:
    - Pairs that fit together get a score based on fill ratio (higher is better)
    - Very tight fits (waste < 5) get an extra bonus to encourage near-perfect complements
    - Pairs that don't fit get a minimal value (1e-9)
    
    Args:
        demands: array-like of item sizes (integers from 20 to 99 passed as floats)
        capacity: bin capacity (150)
    
    Returns:
        n x n matrix where entry [i, j] is the attractiveness of placing item j after item i
    """
    demands = np.asarray(demands, dtype=np.float64)
    n = len(demands)
    
    # Compute pair sums: pair[i, j] = demands[i] + demands[j]
    pair = demands[:, None] + demands[None, :]
    
    # Initialize with minimal value for non-fitting pairs
    matrix = np.full((n, n), 1e-9, dtype=np.float64)
    
    # Mask of fitting pairs
    fits = pair <= capacity
    
    # For fitting pairs, compute waste = capacity - pair
    waste = capacity - pair
    
    # Base score: fill ratio (pair / capacity), raised to a power to sharpen preference
    # Power of 15 makes high fill ratios much more attractive
    fill_ratio = np.where(fits, pair / capacity, 0.0)
    base_score = fill_ratio ** 15
    
    # Tight-fit bonus: for pairs with very little waste, add a bonus
    # This gives near-perfect complements a distinct advantage
    tight_mask = fits & (waste < 5)
    tight_bonus = np.where(tight_mask, 2.0 / (waste + 1.0), 0.0)
    
    # Combine base score and tight bonus
    matrix = np.where(fits, base_score + tight_bonus, 1e-9)
    
    # Ensure diagonal (item with itself) is handled correctly
    # An item can follow itself only if 2*item <= capacity
    for i in range(n):
        if 2.0 * demands[i] <= capacity:
            w = capacity - 2.0 * demands[i]
            fr = (2.0 * demands[i]) / capacity
            score = fr ** 15
            if w < 5:
                score += 2.0 / (w + 1.0)
            matrix[i, i] = score
        else:
            matrix[i, i] = 1e-9
    
    # Ensure no non-finite values
    matrix = np.where(np.isfinite(matrix), matrix, 1e-9)
    
    return matrix
# EVOLVE-BLOCK-END
