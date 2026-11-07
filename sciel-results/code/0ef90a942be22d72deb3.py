def construct():
    # The Fano plane on vertices 0..6 is the unique (up to isomorphism)
    # 3-uniform hypergraph with 7 edges that is not 2-colorable.
    # It is the minimum edge count for a 3-chromatic 3-uniform hypergraph.
    # Vertices 7 and 8 are isolated.
    
    fano_edges = (
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    )

    edges = [list(e) for e in fano_edges]
    
    # Coloring:
    # 0:0, 1:1, 2:2, 3:1, 4:2, 5:0, 6:1
    # 7:0, 8:0
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    return {
        'edges': edges,
        'colors': colors
    }
