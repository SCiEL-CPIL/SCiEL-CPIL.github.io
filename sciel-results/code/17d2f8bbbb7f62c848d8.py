"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Constructs a complete bipartite graph K_{8,8} on 16 vertices.
    This graph has diameter 2 and every edge is critical for maintaining diameter 2.
    Number of edges: 64.
    """
    n = 16
    half = n // 2
    edges = []
    
    # Partition 0: 0 to half-1
    # Partition 1: half to n-1
    for u in range(half):
        for v in range(half, n):
            edges.append([u, v])
            
    return edges
# EVOLVE-BLOCK-END
