"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 16}

def construct():
    # The complete bipartite graph K_{8,8} is the optimal solution.
    # It has 64 edges, is triangle-free (so every edge is critical for diameter 2),
    # and by Mantel's theorem no triangle-free graph on 16 vertices can have more than 64 edges.
    # Any graph with more edges must contain a triangle, which introduces redundant edges
    # for maintaining diameter 2, failing the edge-criticality requirement.
    
    n = P['n']
    half = n // 2
    partition_a = list(range(half))
    partition_b = list(range(half, n))
    
    edges = []
    for u in partition_a:
        for v in partition_b:
            edges.append([u, v])
            
    return edges
# EVOLVE-BLOCK-END
