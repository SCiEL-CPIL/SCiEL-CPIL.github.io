"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    n = 16
    # A star graph K_{1, n-1} has diameter 2.
    # Removing any edge disconnects a leaf from the center,
    # thus satisfying the criticality condition (disconnection).
    # Edge count: n - 1 = 15.
    edges = []
    for v in range(1, n):
        edges.append([0, v])
    return edges
# EVOLVE-BLOCK-END
