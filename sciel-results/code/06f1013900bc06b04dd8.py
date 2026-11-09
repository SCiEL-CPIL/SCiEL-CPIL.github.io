def construct():
    # Fano plane: 7 vertices (0..6), 7 edges
    # Vertices 7 and 8 are unused in edges but included in the color array.
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
    # Coloring: 0->0, 1->1, 2->2, 3->1, 4->2, 5->0, 6->1
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    return {
        'edges': edges,
        'colors': colors
    }
