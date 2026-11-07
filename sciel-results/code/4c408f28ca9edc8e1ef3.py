"""
Constructs the complete bipartite graph K_{8,8} on vertices 0..15.
This graph is triangle-free, has diameter 2, and is edge-critical for diameter 2.
It has the maximum possible number of edges (64) for a triangle-free graph on 16 vertices.
"""

def construct():
    """
    Returns a list of edges [u, v] for the graph K_{8,8}.
    
    Vertices are 0..7 (Partition A) and 8..15 (Partition B).
    Every vertex in A is connected to every vertex in B.
    """
    n = 16
    mid = n // 2
    
    edges = []
    for u in range(mid):
        for v in range(mid, n):
            # Ensure types are Python int
            edges.append([int(u), int(v)])
            
    return edges
