"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

N = 16

def construct():
    """
    Constructs a simple undirected graph on vertices 0..15 with diameter exactly 2,
    where every edge is essential.
    
    The graph used is the complete bipartite graph K_{8,8}.
    Vertices 0..7 form partition A, and 8..15 form partition B.
    Edges exist between every pair (u, v) with u in A and v in B.
    
    Properties:
    1. Diameter is 2. (Any u in A, v in B are adjacent or distance 2 via another node).
       Wait, in K_{8,8}, any u in A and v in B are adjacent. 
       Distance between two nodes in A is 2 (via any node in B).
       Distance between two nodes in B is 2 (via any node in A).
       So diameter is 2.
    2. Every edge is essential.
       Consider an edge (u, v) with u in A, v in B.
       Removing it disconnects u and v in the sense that d(u, v) becomes 3.
       (u is connected to B\{v}, which is connected to A\{u}, which is connected to v? No.)
       Path u - w - x - v. Length 3.
       So diameter becomes 3.
       Thus, the edge is essential.
    
    Returns:
        List of edges as [u, v] lists.
    """
    edges = []
    # Partition A: 0..7
    # Partition B: 8..15
    for u in range(8):
        for v in range(8, 16):
            edges.append([u, v])
            
    # Ensure types are Python int
    result = []
    for e in edges:
        result.append([int(e[0]), int(e[1])])
        
    return result
# EVOLVE-BLOCK-END
