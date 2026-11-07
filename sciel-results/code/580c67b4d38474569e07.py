"""
Constructs a complete bipartite graph K_{8,8} on vertices 0..15.
This graph is triangle-free and has diameter 2.
In a triangle-free graph with diameter 2, every edge is critical:
removing any edge (u, v) eliminates the only path of length 1 between u and v.
Since there are no triangles, there are no paths of length 2 between u and v.
Thus, the distance between u and v becomes at least 3 (or infinity if disconnected, but K_{8,8} remains connected).
This increases the diameter from 2 to at least 3.
The number of edges is 8 * 8 = 64, which is the maximum possible for a triangle-free graph on 16 vertices (Mantel's Theorem).
"""

def construct():
    """
    Returns a list of edges [u, v] for a complete bipartite graph K_{8,8}.
    Vertices are 0..15.
    Partition A: 0..7
    Partition B: 8..15
    """
    n = 16
    mid = n // 2
    set_a = list(range(mid))
    set_b = list(range(mid, n))
    
    edges = []
    for u in set_a:
        for v in set_b:
            edges.append([u, v])
            
    return edges
