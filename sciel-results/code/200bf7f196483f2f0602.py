def construct():
    n = 16
    half = n // 2
    # Partition vertices into two sets of size 8
    set_a = list(range(half))
    set_b = list(range(half, n))
    
    edges = []
    # Construct complete bipartite graph K_{8,8}
    # This is the unique (up to isomorphism) maximum edge triangle-free 
    # diameter-2 graph on 16 vertices.
    for u in set_a:
        for v in set_b:
            edges.append([u, v])
            
    return edges
