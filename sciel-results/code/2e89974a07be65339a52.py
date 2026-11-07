def construct():
    # The Fano plane on vertices 0-6 is the smallest 3-chromatic 3-uniform hypergraph.
    # It has 7 edges.
    # Vertices 7 and 8 are isolated.
    
    edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    
    # A valid 3-coloring for the Fano plane:
    # c(0)=0, c(1)=1, c(2)=2, c(3)=1, c(4)=2, c(5)=0, c(6)=1
    # Vertices 7, 8 are isolated, assigned color 0.
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    return {
        'edges': edges,
        'colors': colors
    }
