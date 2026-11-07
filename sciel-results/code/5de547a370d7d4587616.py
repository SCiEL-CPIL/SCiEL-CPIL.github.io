"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""

# EVOLVE-BLOCK-START

def construct():
    # The Fano plane is a 3-uniform hypergraph on 7 vertices with 7 edges
    # that is not 2-colorable (chromatic number 3).
    # We embed it in 9 vertices (vertices 7 and 8 are isolated).
    # This is the minimum number of edges for a 3-chromatic 3-uniform hypergraph.
    
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # A valid 3-coloring of the Fano plane
    # Colors are 0, 1, 2
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Ensure types are exactly Python int and list
    edges_out = [list(e) for e in edges]
    colors_out = [int(c) for c in colors]
    
    return {'edges': edges_out, 'colors': colors_out}

# EVOLVE-BLOCK-END
