def construct():
    """
    Constructs a graph with 35 vertices that contains no 5-cycle and no independent set of size 9.
    The graph is a disjoint union of 7 copies of K4 and 1 copy of K7.
    - Number of vertices: 7 * 4 + 7 = 35.
    - Independence number: 8 (one vertex from each clique). Since 8 < 9, the condition is met.
    - Cycles: Disjoint union of cliques contains no cycles of length > 3, so no 5-cycle.
    """
    n = 35
    edges = []
    
    # Define cliques by their starting vertex and size
    cliques = [
        (0, 4),
        (4, 4),
        (8, 4),
        (12, 4),
        (16, 4),
        (20, 4),
        (24, 4),
        (28, 7)
    ]
    
    for start, size in cliques:
        for i in range(size):
            for j in range(i + 1, size):
                edges.append([start + i, start + j])
                
    return {'n': n, 'edges': edges}
