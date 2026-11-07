"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Constructs a dense graph on 16 vertices with diameter 2 where every edge is essential.
    The optimal graph is K_{8,8}, which is triangle-free and has 64 edges.
    """
    n = 16
    half = n // 2
    
    # Generate edges for K_{8,8}
    # Partition 1: 0 to 7
    # Partition 2: 8 to 15
    edges = []
    for i in range(half):
        for j in range(half, n):
            edges.append([i, j])
            
    return edges
# EVOLVE-BLOCK-END
