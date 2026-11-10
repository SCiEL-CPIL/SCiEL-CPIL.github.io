def construct():
    """
    Construct a graph on 16 vertices (0..15) with diameter exactly 2,
    where deleting any single edge increases the diameter above 2 or disconnects the graph.
    Maximize the number of edges.

    The optimal graph is the complete bipartite graph K_{8,8}.
    Vertices 0..7 in set A, vertices 8..15 in set B.
    This graph is triangle-free, which is a necessary condition for edge-criticality
    in a diameter-2 graph. By Turan's theorem, any graph on 16 vertices with more than
    64 edges must contain a triangle, making some edge non-critical. Thus K_{8,8} is optimal.
    """
    edges = []
    for u in range(8):
        for v in range(8, 16):
            edges.append([u, v])
    return edges
