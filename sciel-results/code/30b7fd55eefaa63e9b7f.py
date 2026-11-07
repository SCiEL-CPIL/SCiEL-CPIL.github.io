def construct():
    """
    Constructs the Fano plane (PG(2,2)) as a 3-uniform hypergraph on 9 vertices.
    The 7 lines of the Fano plane are the minimum-edge 3-chromatic 3-uniform
    hypergraph. Vertices 7 and 8 are isolated (any color works).
    """
    # Fano plane lines as triples on vertices 0..6
    edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]

    # A valid 3-coloring: each edge contains all three colors
    # Vertex: 0  1  2  3  4  5  6  7  8
    # Color:  0  1  2  1  2  0  1  0  0
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]

    return {
        'edges': edges,
        'colors': colors
    }
