"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start = time.time()
    deadline = start + 60.0

    n = 16
    half = n >> 1

    # Build K_{8,8}: the unique (up to iso) triangle-free graph with max edges = n^2/4.
    # Mantel's theorem guarantees any graph with > n^2/4 edges has a triangle.
    # A triangle makes its edges non-critical for diameter-2 (length-2 path remains).
    # Hence edge-critical diameter-2 graphs must be triangle-free, bounding edges at 64.
    # K_{8,8} achieves 64 and is edge-critical: bipartite => no common neighbors across
    # partitions => removing any edge leaves the two endpoints at distance 3.

    result = []
    for i in range(half):
        for j in range(half, n):
            result.append([i, j])

    # Deadline guard (instantaneous in practice, but satisfies contract)
    if time.time() > deadline:
        return result

    return result
# EVOLVE-BLOCK-END
