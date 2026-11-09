"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    n = len(demands)
    pair = demands[:, None] + demands[None, :]
    mask = pair <= capacity
    ratio = np.where(mask, pair / capacity, 0.0)
    # Strong power-law to heavily favor tight fits (pairs close to capacity)
    # Exponent 16 creates very strong preference for near-full bins
    heuristic = np.where(mask, ratio ** 16, 1e-9)
    return heuristic
# EVOLVE-BLOCK-END
