"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # Fano plane: 7 edges on 7 vertices (0..6), 3-chromatic 3-uniform hypergraph
    # Vertices 7,8 are isolated (color 0)
    edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5),
    ]
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    return {'edges': [list(e) for e in edges], 'colors': colors}

# EVOLVE-BLOCK-END
