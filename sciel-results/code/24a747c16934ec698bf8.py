"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""

# EVOLVE-BLOCK-START

def construct():
    # The minimum number of edges in a 3-uniform hypergraph that is 
    # not 2-colorable (3-chromatic) is 7. The unique such hypergraph 
    # on 7 vertices is the Fano plane. 
    # We construct the Fano plane on vertices 0..6. Vertices 7 and 8 
    # are isolated and do not affect the chromatic number.
    
    # Fano plane edges (triples)
    # This is a standard representation of the 7 lines of the Fano plane.
    fano_edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # A valid 3-coloring for the Fano plane.
    # Colors are 0, 1, 2.
    # We verify no edge is monochromatic.
    # v0:0, v1:1, v2:2
    # v3:1, v4:2, v5:0
    # v6:1
    # v7:0, v8:0 (isolated, arbitrary)
    
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Quick sanity check: ensure no edge is monochromatic in our coloring.
    # This prevents returning an invalid result due to a typo in colors/edges.
    for e in fano_edges:
        c0 = colors[e[0]]
        c1 = colors[e[1]]
        c2 = colors[e[2]]
        if c0 == c1 == c2:
            # If this happens, the coloring is invalid. 
            # However, for the known Fano plane coloring, this should never happen.
            # We raise an error to signal a bug in the hardcoded data.
            raise ValueError("Invalid coloring: monochromatic edge detected")
            
    # The Fano plane is known to be 3-chromatic (not 2-colorable).
    # The evaluator will verify this.
    
    # Ensure types are exactly Python int and list
    edges = [list(e) for e in fano_edges]
    colors_out = [int(c) for c in colors]
    
    return {
        'edges': edges,
        'colors': colors_out
    }

# EVOLVE-BLOCK-END
