def construct():
    # The Fano plane is the minimal 3-uniform hypergraph that is not 2-colorable.
    # It has 7 vertices and 7 edges.
    # We embed it in 9 vertices by leaving vertices 7 and 8 isolated.
    
    # Fano plane edges (standard cyclic construction or standard lines)
    # Vertices 0..6
    fano_edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # A valid 3-coloring for the Fano plane
    # Colors for vertices 0..6
    c = [0, 1, 2, 1, 2, 0, 1]
    # Vertices 7, 8 can be any color, e.g., 0
    c.append(0)
    c.append(0)
    
    return {
        'edges': [list(e) for e in fano_edges],
        'colors': c
    }
