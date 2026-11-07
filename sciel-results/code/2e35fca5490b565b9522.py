def construct():
    n = 16
    # Complete bipartite graph K_{8,8}
    # This is a triangle-free graph with diameter 2.
    # In any triangle-free graph, every edge is essential for diameter 2:
    # if an edge (u,v) is removed, since there is no triangle, there is no
    # alternative path of length 2 between u and v, so the distance becomes
    # at least 3 (or infinity if disconnected), violating the diameter-2 property.
    edges = []
    A = list(range(8))
    B = list(range(8, 16))
    
    for u in A:
        for v in B:
            edges.append([int(u), int(v)])
    
    return edges
