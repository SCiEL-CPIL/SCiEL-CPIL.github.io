def construct():
    """
    Constructs the complete bipartite graph K_{8,8} on 16 vertices.
    This graph has 64 edges, which is the maximum possible for a triangle-free graph
    on 16 vertices (Mantel's Theorem). Since any triangle would render an edge
    non-essential for maintaining diameter 2, K_{8,8} is the optimal solution.
    """
    edges = []
    for u in range(8):
        for v in range(8, 16):
            edges.append([u, v])
    return edges
