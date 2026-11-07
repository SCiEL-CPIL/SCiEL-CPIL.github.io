"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix):
    # Use a higher exponent to make the heuristic more greedy (favor short edges more strongly)
    # This helps the ACO avoid long, suboptimal jumps in the early iterations.
    return 1.0 / (distance_matrix + 1e-9)**3
# EVOLVE-BLOCK-END
