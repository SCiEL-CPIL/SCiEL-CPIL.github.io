"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # The Fano plane is the minimum 3-chromatic 3-uniform hypergraph with 7 edges.
    # It is defined on 7 vertices. We use vertices 0..6 and leave 7,8 unused.
    # Edges of the Fano plane:
    edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    
    # A valid 3-coloring for the Fano plane vertices 0..6:
    # 0:0, 1:1, 2:2, 3:1, 4:2, 5:0, 6:0
    # Check:
    # (0,1,2) -> 0,1,2 (non-mono)
    # (0,3,4) -> 0,1,2 (non-mono)
    # (0,5,6) -> 0,0,0 (MONOCHROMATIC!) -> INVALID
    
    # Let's re-verify the coloring.
    # Fano plane edges:
    # E1: 0,1,2
    # E2: 0,3,4
    # E3: 0,5,6
    # E4: 1,3,5
    # E5: 1,4,6
    # E6: 2,3,6
    # E7: 2,4,5
    
    # Try coloring:
    # c0 = 0
    # c1 = 1
    # c2 = 2
    # c3 = 1
    # c4 = 2
    # c5 = 0
    # c6 = ?
    
    # E1 (0,1,2): 0,1,2 -> OK
    # E2 (0,3,4): 0,1,2 -> OK
    # E3 (0,5,6): 0,0,c6. Need c6 != 0. Let c6 = 1.
    # E4 (1,3,5): 1,1,0 -> OK
    # E5 (1,4,6): 1,2,1 -> OK
    # E6 (2,3,6): 2,1,1 -> OK
    # E7 (2,4,5): 2,2,0 -> OK
    
    # So coloring [0, 1, 2, 1, 2, 0, 1] works for 0..6.
    # Vertices 7 and 8 can be 0.
    
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    return {
        'edges': [list(e) for e in edges],
        'colors': colors
    }

# EVOLVE-BLOCK-END
