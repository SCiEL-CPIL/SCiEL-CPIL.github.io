"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Constructs a graph on 16 vertices (0..15) with diameter exactly 2.
    Every edge is essential: deleting any edge increases diameter or disconnects.
    Maximize edge count.
    
    Solution: Complete bipartite graph K(8,8).
    - Triangle-free.
    - Diameter 2.
    - Removing any edge (u,v) makes u and v distance 3 (since no common neighbors in bipartite graph for cross-partition edges? 
      Wait. In K(8,8), u in A, v in B.
      N(u) = B. N(v) = A.
      N(u) intersection N(v) is empty.
      So no path of length 2 between u and v.
      Shortest path becomes length 3: u - b' - a' - v.
      So diameter becomes at least 3.
    - Edge count: 8 * 8 = 64.
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
