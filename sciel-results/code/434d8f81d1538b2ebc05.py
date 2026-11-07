"""Constructs a maximal triangle-free graph with diameter 2 on 16 vertices."""

def construct():
    """
    Returns the edges of K_{8,8}, which is the unique triangle-free graph 
    on 16 vertices with the maximum number of edges (64) and diameter 2.
    In this graph, every edge is essential for maintaining diameter 2 
    (removing any edge increases the distance between its endpoints to 3).
    """
    n = 16
    half = n // 2
    
    # Partitions: 0..7 and 8..15
    # Edges connect every vertex in the first partition to every vertex in the second.
    edges = []
    for u in range(half):
        for v in range(half, n):
            edges.append([int(u), int(v)])
            
    return edges
