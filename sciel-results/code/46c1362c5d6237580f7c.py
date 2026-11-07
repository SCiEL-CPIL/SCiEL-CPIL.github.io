def construct():
    # Fano plane: the unique 3-uniform hypergraph on 7 vertices
    # with chromatic number 3 (not 2-colorable). 7 edges is the
    # minimum possible for a 3-chromatic 3-uniform hypergraph.
    # Vertices 0..6 form the Fano plane; 7, 8 are isolated.
    edges = (
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5),
    )
    colors = (0, 1, 2, 1, 2, 0, 1, 0, 0)
    return {'edges': edges, 'colors': colors}
