def construct():
    """
    Constructs a 3-uniform hypergraph on 9 vertices (0..8) that is 
    3-chromatic (not 2-colorable) with the minimum number of edges.
    
    The Fano plane (7 edges) is the smallest 3-uniform non-2-colorable hypergraph.
    """
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # Valid 3-coloring for the Fano plane
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    return {'edges': edges, 'colors': colors}
