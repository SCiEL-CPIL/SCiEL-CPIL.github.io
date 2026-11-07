"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    total_weight = weights.sum(axis=0)
    max_prize = np.max(prizes, axis=0)
    return max_prize / (total_weight + 1e-9)
# EVOLVE-BLOCK-END
