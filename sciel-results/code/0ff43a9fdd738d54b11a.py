"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    n = len(demands)
    pair = demands[:, None] + demands[None, :]
    # For pairs that fit, reward strongly based on how close to capacity they are
    # Use a power function to amplify the preference for tight fits
    ratio = pair / capacity
    # Only consider pairs that fit (ratio <= 1)
    mask = pair <= capacity
    # For fitting pairs, use (ratio)^4 to strongly favor tight fits
    # For non-fitting pairs, use 1e-9
    heuristic = np.where(mask, ratio ** 4, 1e-9)
    return heuristic
# EVOLVE-BLOCK-END
