"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # The Fano plane is the unique minimal 3-chromatic 3-uniform hypergraph.
    # It has 7 vertices and 7 edges.
    # We map its vertices to 0..6. Vertices 7 and 8 are isolated.
    # 7 is the theoretical minimum number of edges for a 3-chromatic 3-uniform hypergraph.
    
    edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    
    # Valid 3-coloring:
    # 0:0, 1:1, 2:2, 3:0, 4:1, 5:0, 6:2
    # Isolated vertices 7,8 assigned color 0.
    colors = [0, 1, 2, 0, 1, 0, 2, 0, 0]
    
    return {'edges': edges, 'colors': colors}

# EVOLVE-BLOCK-END
