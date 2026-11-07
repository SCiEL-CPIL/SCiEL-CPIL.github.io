"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    """
    Compute an item-by-item attractiveness matrix for placing item j right
    after item i in the same bin.
    
    Strategy: Use a high-power fill ratio to sharply prefer pairs that nearly
    fill the bin, plus a tight-fit bonus for pairs with very little waste.
    This guides ants to close bins with minimal leftover space.
    """
    demands = np.asarray(demands, dtype=np.float64)
    n = len(demands)
    
    # Pair sums
    pair = demands[:, None] + demands[None, :]
    
    # Valid mask: pair must fit in the bin
    valid = pair <= capacity
    
    # Fill ratio: how much of the capacity the pair uses
    fill_ratio = np.where(valid, pair / capacity, 0.0)
    
    # High power exponent sharply rewards high fill ratios
    # (pair/capacity)^15: 0.9^15≈0.206, 0.95^15≈0.463, 1.0^15=1.0
    fill_power = fill_ratio ** 15
    
    # Tight-fit bonus: give near-perfect complements (waste < 5) a distinct advantage
    # waste = capacity - pair_sum, for valid pairs
    waste = np.where(valid, capacity - pair, 0.0)
    # 2/(waste+1) gives: waste=0 -> 2.0, waste=1 -> 1.0, waste=2 -> 0.67, waste=4 -> 0.4
    tight_bonus = np.where((valid) & (waste < 5), 2.0 / (waste + 1.0), 0.0)
    
    # Combine: primary fill signal + tight-fit bonus
    heuristic = np.where(valid, fill_power + tight_bonus, 0.0)
    
    # Ensure all entries are positive (minimum 1e-9)
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
