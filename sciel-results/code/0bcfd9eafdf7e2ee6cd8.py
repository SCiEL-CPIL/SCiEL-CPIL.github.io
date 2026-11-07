def construct():
    """
    Constructs a graph on 12 vertices (0..11) consisting of two disjoint K_{3,3} components.
    
    Properties:
    - Undirected, simple graph.
    - Minimum degree is 3.
    - Triangle-free (bipartite).
    - Simple cycles in K_{3,3} have lengths 4 and 6.
    - Distinct cycle lengths: {4, 6}. Count = 2.
    """
    edges = []
    
    # Component 1: Vertices 0, 1, 2 (Part A) and 3, 4, 5 (Part B)
    part_a1 = [0, 1, 2]
    part_b1 = [3, 4, 5]
    for u in part_a1:
        for v in part_b1:
            edges.append([u, v])
            
    # Component 2: Vertices 6, 7, 8 (Part C) and 9, 10, 11 (Part D)
    part_a2 = [6, 7, 8]
    part_b2 = [9, 10, 11]
    for u in part_a2:
        for v in part_b2:
            edges.append([u, v])
            
    return edges
