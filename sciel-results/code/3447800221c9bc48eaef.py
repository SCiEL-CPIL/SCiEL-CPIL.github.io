def construct():
    """
    Construct a simple undirected graph on 16 vertices (0..15) with diameter exactly two,
    where every edge is critical (deleting it increases diameter > 2 or disconnects it).
    Maximize edge count.
    
    The optimal solution is the complete bipartite graph K_{8,8}.
    - It has 64 edges, which is the maximum possible for a triangle-free graph on 16 vertices (Mantel's Theorem).
    - It is triangle-free, ensuring that every edge is critical (removing any edge (u,v) increases the distance between u and v to 3, thus increasing the diameter to at least 3).
    - It has diameter 2.
    """
    n = 16
    half = n // 2
    edges = []
    
    # Partition 1: 0 .. half-1
    # Partition 2: half .. n-1
    for u in range(half):
        for v in range(half, n):
            edges.append([int(u), int(v)])
            
    return edges
