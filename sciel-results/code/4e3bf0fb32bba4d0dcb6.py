"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # The Fano plane is the smallest 3-uniform hypergraph with chromatic number 3.
    # It has 7 edges and 7 vertices.
    # We embed it on vertices 0..6 and leave 7, 8 isolated.
    # This results in a hypergraph on 9 vertices with 7 edges.
    
    edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    
    # A valid 3-coloring for the Fano plane vertices 0..6
    # v0: 0, v1: 1, v2: 2, v3: 1, v4: 0, v5: 0, v6: 1
    # Isolated vertices 7, 8 can be any color, say 0.
    colors = [0, 1, 2, 1, 0, 0, 1, 0, 0]
    
    return {
        'edges': [list(e) for e in edges],
        'colors': colors
    }

# EVOLVE-BLOCK-END
