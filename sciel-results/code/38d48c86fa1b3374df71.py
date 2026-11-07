"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix):
    # Use inverse cubed distance to strongly emphasize short edges.
    # This guides ants to build compact tours early, improving convergence.
    # Add a small epsilon to avoid division by zero on the diagonal.
    return 1.0 / (distance_matrix ** 3 + 1e-9)
# EVOLVE-BLOCK-END
