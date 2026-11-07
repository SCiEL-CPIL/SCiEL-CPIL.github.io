"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # The Fano plane is the unique minimal 3-uniform hypergraph that is 3-chromatic.
    # It has 7 edges on 7 vertices.
    # We embed it in 9 vertices (0..8) by leaving vertices 7 and 8 isolated.
    # This minimizes the edge count to 7.
    
    # Fano plane edges on vertices 0..6
    fano_edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    
    # Convert to lists as required by "sequences must be list or tuple" (lists are safer for JSON-like structures if needed, though tuple is fine, task says list or tuple)
    # The task says "sequences must be list or tuple".
    edges = [list(e) for e in fano_edges]
    
    # A valid 3-coloring for vertices 0..8
    # Vertices 0..6 must be colored such that no Fano edge is monochromatic.
    # Vertices 7,8 can be any color.
    # Coloring:
    # 0:0, 1:1, 2:2
    # 3:0, 4:1, 5:2
    # 6:0
    # 7:0, 8:0
    colors = [0, 1, 2, 0, 1, 2, 0, 0, 0]
    
    # Verify constraints just in case
    for e in edges:
        c0 = colors[e[0]]
        c1 = colors[e[1]]
        c2 = colors[e[2]]
        if c0 == c1 == c2:
            # This should not happen with the fixed coloring above
            # But if it did, we'd need to fix it. 
            # For robustness, we could run a small check.
            pass
            
    return {'edges': edges, 'colors': colors}
# EVOLVE-BLOCK-END
