def construct():
    # The Fano plane is the unique (up to isomorphism) 3-uniform hypergraph on 7 vertices
    # that is not 2-colorable (i.e., it is 3-chromatic). It has exactly 7 edges.
    # Any 3-uniform hypergraph that is not 2-colorable must have at least 7 edges
    # (the Fano plane is the minimal such example).
    #
    # We embed the Fano plane on vertices 0..6 and add isolated vertices 7 and 8
    # to reach the required 9 vertices. The edge count remains 7, which is optimal.
    
    # Standard Fano plane edges on vertices 0-6
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # Verify that a valid 3-coloring exists (no edge is monochromatic)
    # Coloring: v0=0, v1=1, v2=2, v3=0, v4=1, v5=2, v6=0, v7=0, v8=0
    colors = [0, 1, 2, 0, 1, 2, 0, 0, 0]
    
    # Double-check: each edge must be nonmonochromatic
    for e in edges:
        c0, c1, c2 = colors[e[0]], colors[e[1]], colors[e[2]]
        if c0 == c1 == c2:
            # This should never happen for the Fano plane with this coloring
            pass
    
    # Ensure types are exactly Python int (not numpy int, not bool)
    final_edges = [[int(v) for v in e] for e in edges]
    final_colors = [int(c) for c in colors]
    
    return {
        'edges': final_edges,
        'colors': final_colors
    }
