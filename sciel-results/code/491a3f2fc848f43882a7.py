def construct():
    # The Fano plane is the unique 3-uniform hypergraph with the minimum
    # number of edges (7) that is 3-chromatic.
    # We place it on vertices 0..6; vertices 7 and 8 are isolated.
    # A valid 3-coloring is provided; the evaluator verifies no 2-coloring exists.
    
    edges = [
        (0, 1, 3),
        (0, 2, 4),
        (0, 5, 6),
        (1, 2, 5),
        (1, 4, 6),
        (2, 3, 6),
        (3, 4, 5)
    ]
    
    # 3-coloring: vertices 0,6 -> color 0; 1,4,5 -> color 1; 2,3 -> color 2; 7,8 -> color 0
    colors = [0, 1, 2, 2, 1, 1, 0, 0, 0]
    
    return {
        'edges': list(edges),
        'colors': list(colors)
    }
