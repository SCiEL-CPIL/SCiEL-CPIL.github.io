"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    n = 16
    half = n // 2
    # Part A: 0..7, Part B: 8..15
    edges = []
    for u in range(half):
        for v in range(half, n):
            edges.append([u, v])
    # Return as list of [int, int]
    return edges
# EVOLVE-BLOCK-END
