"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    n = len(demands)
    pair = demands[:, None] + demands[None, :]
    # Favor pairs that fill the bin closely (tight packing)
    # Higher score when pair sum is close to capacity but still fits
    # Use a Gaussian-like bump centered at capacity to reward tight fits
    # Also penalize very small pairs (waste space)
    valid = pair <= capacity
    # Reward: closer to capacity is better, but avoid exactly 0
    # Use (pair/capacity)^2 to strongly reward large pairs that fit
    # Combined with a small bonus for moderate pairs
    score = np.where(valid, (pair / capacity) ** 2, 0.0)
    # Ensure all entries are positive (minimum 1e-9)
    score = np.maximum(score, 1e-9)
    return score
# EVOLVE-BLOCK-END
