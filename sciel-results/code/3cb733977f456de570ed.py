"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 16}

def construct():
    """
    Return edges of K_{8,8} on vertices 0..15.
    
    K_{8,8} has 64 edges, diameter exactly 2, and every edge is critical
    (removing any edge increases the distance between its endpoints to 3,
    thus increasing the diameter above 2). This is an extremal example
    of a minimally diameter-2 graph with maximum edge count (by Mantel's Theorem).
    """
    n = P['n']
    half = n // 2  # 8
    
    # K_{8,8}: vertices 0-7 in one partition, 8-15 in the other
    edges = []
    for u in range(half):
        for v in range(half, n):
            edges.append([int(u), int(v)])
    
    return edges
# EVOLVE-BLOCK-END
