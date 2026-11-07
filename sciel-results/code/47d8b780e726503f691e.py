"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    # prizes: (5, 100), weights: (5, 100)
    # Total prize per item
    total_prize = prizes.sum(axis=0)  # (100,)
    # Total weight per item across all constraints
    total_weight = weights.sum(axis=0)  # (100,)
    # Heuristic: prize density (higher is more attractive)
    heuristic = total_prize / (total_weight + 1e-9)
    return heuristic
# EVOLVE-BLOCK-END
