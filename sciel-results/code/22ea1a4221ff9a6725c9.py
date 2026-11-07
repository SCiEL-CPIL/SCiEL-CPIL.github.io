"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    # Use inverse squared distance to emphasize short connections more strongly.
    # This creates a steeper gradient for the ant colony, encouraging tighter clusters.
    d = distance_matrix
    # Avoid division by zero
    d_safe = d + 1e-9
    eta = 1.0 / (d_safe ** 2)
    return eta
# EVOLVE-BLOCK-END
