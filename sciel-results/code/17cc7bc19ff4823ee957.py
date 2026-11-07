"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Constructs a graph on 16 vertices (0..15) with diameter exactly 2.
    Every edge is essential: deleting any edge increases diameter or disconnects.
    Maximize edge count.
    
    Solution: Complete bipartite graph K(8,8).
    - This is the maximum edge count for an edge-critical diameter-2 graph on 16 vertices.
    - Any graph with more edges would contain a triangle or have non-critical edges.
    - Edge count: 64.
    """
    n = 16
    # Partition vertices into two sets of 8
    A = list(range(0, 8))
    B = list(range(8, 16))
    
    edges = []
    for u in A:
        for v in B:
            edges.append([u, v])
            
    return edges
# EVOLVE-BLOCK-END
