"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.monotonic() + 55.0  # safety margin under 60s evaluator limit
    n = 16
    half = n // 2
    edges = []
    # K_{8,8}: partition A = 0..7, partition B = 8..15
    # Triangle-free, diameter exactly 2, every edge critical.
    # Mantel's theorem gives 64 as the maximum triangle-free edge count on 16 vertices.
    for u in range(half):
        for v in range(half, n):
            edges.append([u, v])
    return edges
# EVOLVE-BLOCK-END
