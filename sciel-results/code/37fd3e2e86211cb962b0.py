"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # The Fano plane is the unique 3-uniform hypergraph with 7 edges that is 3-chromatic.
    # Any 3-uniform hypergraph with <= 6 edges is 2-colorable.
    # Thus, 7 is the minimum possible raw value (edge count).
    # Vertices 7 and 8 are isolated, which is valid for a hypergraph on 0..8.
    
    edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    
    # A valid 3-coloring of the Fano plane where no edge is monochromatic.
    # Vertices 7 and 8 can be assigned any color, e.g., 0.
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Ensure strict type compliance: Python int for colors, list of lists for edges.
    edges_list = [list(e) for e in edges]
    colors_list = [int(c) for c in colors]
    
    return {'edges': edges_list, 'colors': colors_list}
# EVOLVE-BLOCK-END
