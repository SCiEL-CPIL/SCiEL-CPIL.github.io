"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # The Fano plane is a 3-uniform hypergraph on 7 vertices with 7 edges.
    # It is known to be 3-chromatic (cannot be 2-colored).
    # Vertices 7 and 8 are not part of the core hypergraph, so they can be colored arbitrarily.
    # This achieves the minimum edge count of 7 for a 3-chromatic 3-uniform hypergraph.
    
    edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    
    # A valid 3-coloring for vertices 0..6:
    # 0: 0, 1: 1, 2: 2
    # 3: 1, 4: 2, 5: 0
    # 6: 1
    # Check edges:
    # (0,1,2) -> 0,1,2 (ok)
    # (0,3,4) -> 0,1,2 (ok)
    # (0,5,6) -> 0,0,1 (ok)
    # (1,3,5) -> 1,1,0 (ok)
    # (1,4,6) -> 1,2,1 (ok)
    # (2,3,6) -> 2,1,1 (ok)
    # (2,4,5) -> 2,2,0 (ok)
    
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    return {
        'edges': [list(e) for e in edges],
        'colors': colors
    }

# EVOLVE-BLOCK-END
