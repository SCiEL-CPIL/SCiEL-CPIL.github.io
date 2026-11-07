def construct():
    # Fano plane: 7 vertices, 7 edges. Minimum 3-uniform 3-chromatic hypergraph.
    # Vertices 7-8 are isolated.
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # Valid 3-coloring for Fano plane (vertices 0-6).
    # Isolated vertices 7-8 colored 0.
    colors = [0, 1, 2, 0, 1, 2, 0, 0, 0]
    
    # Ensure types are strictly Python int and list
    edges = [list(e) for e in edges]
    colors = [int(c) for c in colors]
    
    return {'edges': edges, 'colors': colors}
