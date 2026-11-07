def construct():
    # The Fano plane is the smallest 3-uniform hypergraph with chromatic number 3.
    # It has exactly 7 edges and 7 vertices, so 7 edges is the theoretical minimum
    # for any three-chromatic 3-uniform hypergraph.

    edges = (
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5),
    )

    # Valid 3-coloring: no edge is monochromatic.
    colors = (0, 1, 2, 1, 2, 0, 1, 0, 0)

    return {'edges': edges, 'colors': colors}
