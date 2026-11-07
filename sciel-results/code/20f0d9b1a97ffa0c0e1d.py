def construct():
    # The Fano plane on vertices 0..6 is a 3-uniform hypergraph that is
    # not 2-colorable. Its chromatic number is exactly 3.
    # It has 7 edges, which is the minimum possible edge count for any
    # 3-chromatic 3-uniform hypergraph.
    # Vertices 7 and 8 are isolated to satisfy the 9-vertex requirement
    # without increasing the edge count.
    
    fano_edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]

    edges = [list(e) for e in fano_edges]
    
    # Valid 3-coloring where no edge is monochromatic
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    return {
        'edges': edges,
        'colors': colors
    }
