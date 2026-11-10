"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    n = 16
    # The complete bipartite graph K_{8,8} has 64 edges, diameter 2,
    # and every edge is essential (removing any edge increases distance
    # between its endpoints to 3, thus diameter >= 3).
    
    edges = []
    for u in range(8):
        for v in range(8, 16):
            edges.append([u, v])
    
    return edges
# EVOLVE-BLOCK-END
