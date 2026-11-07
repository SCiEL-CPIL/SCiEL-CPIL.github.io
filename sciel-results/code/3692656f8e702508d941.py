"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Constructs a complete bipartite graph K_{8,8} on 16 vertices.
    This graph has diameter 2 and every edge is essential for maintaining
    diameter 2 (removing any edge increases the distance between its
    endpoints to 3, thus increasing the graph diameter to 3).
    The number of edges is 8 * 8 = 64, which is the maximum possible
    for a triangle-free graph on 16 vertices (Mantel's Theorem).
    """
    n = 16
    half = n // 2
    u_set = list(range(half))
    v_set = list(range(half, n))
    
    edges = []
    for u in u_set:
        for v in v_set:
            edges.append([u, v])
            
    return edges
# EVOLVE-BLOCK-END
