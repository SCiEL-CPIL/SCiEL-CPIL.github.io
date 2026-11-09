def construct():
    """
    Constructs a graph on 12 vertices (0..11) with minimum degree 3 and girth >= 4.
    Uses two disjoint K_{3,3} components. Each K_{3,3} has only 4- and 6-cycles.
    Disconnected graphs are allowed. Distinct cycle lengths: {4, 6} = 2.
    """
    edges = []

    # First K_{3,3} on vertices 0..5
    A1 = [0, 1, 2]
    B1 = [3, 4, 5]
    for u in A1:
        for v in B1:
            edges.append([u, v])

    # Second K_{3,3} on vertices 6..11
    A2 = [6, 7, 8]
    B2 = [9, 10, 11]
    for u in A2:
        for v in B2:
            edges.append([u, v])

    return edges
