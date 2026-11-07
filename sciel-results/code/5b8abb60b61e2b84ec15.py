def construct():
    # The minimal 3-chromatic 3-uniform hypergraph is the Fano Plane.
    # It has 7 vertices and 7 edges.
    # We use vertices 0-6 for the Fano Plane and 7-8 are isolated.
    # This minimizes the edge count to 7.
    
    edges = [
        [0, 1, 3],
        [0, 2, 4],
        [0, 5, 6],
        [1, 2, 5],
        [1, 4, 6],
        [2, 3, 6],
        [3, 4, 5]
    ]
    
    # Valid 3-coloring for the Fano Plane vertices (0-6)
    # Isolated vertices (7, 8) can have any color.
    colors = [0, 1, 2, 2, 1, 1, 0, 0, 0]
    
    return {
        'edges': edges,
        'colors': colors
    }
