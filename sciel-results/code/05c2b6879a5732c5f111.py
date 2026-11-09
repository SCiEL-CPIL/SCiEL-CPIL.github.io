"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    """
    Constructs the complete bipartite graph K_{8,8} on 16 vertices.
    
    Properties:
    1. Diameter is exactly 2.
    2. Every edge is critical: Removing any edge (u,v) where u in A, v in B 
       eliminates the direct connection. Since the graph is bipartite, 
       there are no paths of length 2 between u and v (any path must be 
       odd length). The next shortest path is length 3. Thus diameter becomes 3.
    3. Maximized edge count: The graph must be triangle-free for edges to be 
       critical (if u,v have common neighbor w, removing (u,v) leaves path 
       u-w-v of length 2, so diameter doesn't necessarily jump above 2). 
       Mantel's theorem states max edges in triangle-free graph on 16 nodes 
       is 64, achieved by K_{8,8}.
    """
    n = 16
    half = n // 2
    edges = []
    
    # Vertices 0..7 are in partition A
    # Vertices 8..15 are in partition B
    for u in range(half):
        for v in range(half, n):
            edges.append([u, v])
            
    return edges
# EVOLVE-BLOCK-END
