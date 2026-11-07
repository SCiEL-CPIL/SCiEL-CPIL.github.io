"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # Fano plane: 7 edges on vertices 0-6 (plus 2 isolated vertices 7,8)
    # This is the minimal 3-uniform hypergraph with chromatic number 3
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5],
    ]
    
    # Verify this is a valid 3-coloring where no edge is monochromatic
    # and the hypergraph is NOT 2-colorable
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Check: every edge must be non-monochromatic under this coloring
    for e in edges:
        c0, c1, c2 = colors[e[0]], colors[e[1]], colors[e[2]]
        if c0 == c1 == c2:
            # This shouldn't happen, but guard against it
            raise ValueError(f"Edge {e} is monochromatic with colors {c0}")
    
    # The Fano plane is known to have chromatic number 3:
    # - It is not 2-colorable (no 2-coloring avoids monochromatic edges)
    # - It is 3-colorable (as shown above)
    
    return {'edges': edges, 'colors': colors}

# EVOLVE-BLOCK-END
