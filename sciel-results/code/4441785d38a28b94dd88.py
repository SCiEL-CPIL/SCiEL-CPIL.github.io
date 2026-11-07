def construct():
    """
    Constructs a graph on 12 vertices (0-11) consisting of two disjoint
    copies of the complete bipartite graph K_{3,3}.
    
    Properties:
    - Vertices 0, 1, 2 form one partition of the first K_{3,3}.
    - Vertices 3, 4, 5 form the other partition of the first K_{3,3}.
    - Vertices 6, 7, 8 form one partition of the second K_{3,3}.
    - Vertices 9, 10, 11 form the other partition of the second K_{3,3}.
    
    Each vertex has degree 3.
    The graph is triangle-free (bipartite), so girth is 4.
    The simple cycles in K_{3,3} have lengths 4 and 6.
    Thus, the set of distinct simple cycle lengths is {4, 6}, count = 2.
    """
    edges = []
    
    # First K_{3,3} component: {0, 1, 2} x {3, 4, 5}
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.append([u, v])
            
    # Second K_{3,3} component: {6, 7, 8} x {9, 10, 11}
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.append([u, v])
            
    return edges
