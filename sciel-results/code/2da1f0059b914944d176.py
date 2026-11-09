"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # Fano plane: 7 edges on 9 vertices (0..8), 3-chromatic 3-uniform hypergraph.
    # The Fano plane has 7 edges and is the proven minimum for a 3-chromatic
    # 3-uniform hypergraph. We embed it on vertices 0..6 and leave 7,8 unused
    # (they can be assigned any color since no edge touches them).
    
    # Fano plane edges (standard representation):
    fano_edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5),
    ]
    
    # Coloring: assign colors 0, 1, 2 to make every edge non-monochromatic.
    # For the Fano plane, a valid 3-coloring exists. One such coloring:
    # We need to verify this is a proper 3-coloring (no edge is monochromatic).
    # Actually, the Fano plane is 3-chromatic, meaning it CANNOT be 2-colored,
    # but it CAN be 3-colored. We need to provide a 3-coloring where no edge
    # is monochromatic.
    
    # Let's use a known 3-coloring of the Fano plane:
    # Color 0: {0, 1, 2}  -> edge (0,1,2) would be monochromatic! Bad.
    
    # We need a coloring where NO edge is monochromatic. Let me think carefully.
    # The Fano plane has 7 points and 7 lines. We need to assign 3 colors to 9 vertices
    # such that each of the 7 edges has at least two different colors.
    
    # Simple approach: assign colors cyclically or use a known valid coloring.
    # Let's try: vertex i gets color (i % 3) for i in 0..8.
    # Edge (0,1,2): colors 0,1,2 - not monochromatic. Good.
    # Edge (0,3,4): colors 0,0,1 - not monochromatic. Good.
    # Edge (0,5,6): colors 0,2,0 - not monochromatic. Good.
    # Edge (1,3,5): colors 1,0,2 - not monochromatic. Good.
    # Edge (1,4,6): colors 1,1,0 - not monochromatic. Good.
    # Edge (2,3,6): colors 2,0,0 - not monochromatic. Good.
    # Edge (2,4,5): colors 2,1,2 - not monochromatic. Good.
    
    # So colors[i] = i % 3 works for the Fano plane embedded on vertices 0..6.
    # Vertices 7 and 8 can be any color; let's use 0 for simplicity.
    
    colors = [0, 1, 2, 0, 1, 2, 0, 0, 0]
    
    # Verify no edge is monochromatic:
    for edge in fano_edges:
        c0 = colors[edge[0]]
        c1 = colors[edge[1]]
        c2 = colors[edge[2]]
        assert not (c0 == c1 == c2), f"Edge {edge} is monochromatic with colors {c0},{c1},{c2}"
    
    edges = [list(e) for e in fano_edges]
    
    return {'edges': edges, 'colors': colors}

# EVOLVE-BLOCK-END
