"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    # Use the bottleneck constraint (max weight across all constraints)
    # This better captures which constraint limits each item
    max_weight = weights.max(axis=0)
    return prizes / (max_weight + 1e-9)
# EVOLVE-BLOCK-END
