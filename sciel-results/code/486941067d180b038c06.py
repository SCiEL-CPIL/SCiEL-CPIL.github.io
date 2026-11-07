"""Rewritten program with verification logic."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Constructs a graph on 16 vertices with diameter exactly 2, 
    where deleting any single edge increases the diameter.
    Returns a list of edges [u, v].
    
    The optimal graph is K_{8,8} (complete bipartite graph with parts of size 8).
    - Total edges: 8 * 8 = 64.
    - Diameter is 2 (any two vertices in the same part have distance 2 via the other part; any two in different parts have distance 1).
    - Every edge is critical: removing an edge (u,v) where u in A, v in B creates a path of length 3 between u and v (u-a-b-v), so the distance becomes 3, increasing the diameter to 3.
    """
    n = 16
    
    # Part A: 0..7
    # Part B: 8..15
    edges = []
    
    # Generate edges for complete bipartite graph K_{8,8}
    for u in range(8):
        for v in range(8, 16):
            edges.append([u, v])
            
    return edges
# EVOLVE-BLOCK-END
