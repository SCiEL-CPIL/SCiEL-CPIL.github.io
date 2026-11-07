"""Rewritten program: find a 3-chromatic 3-uniform hypergraph on 9 vertices with minimum edges."""
# EVOLVE-BLOCK-START

def construct():
    """
    Returns a 3-chromatic 3-uniform hypergraph on vertices 0..8 with the minimum number of edges.
    The minimum number of edges for a 3-chromatic 3-uniform hypergraph is 7 (the Fano plane).
    We construct the Fano plane on vertices 0..6 and leave vertices 7..8 isolated.
    """
    # Fano plane edges
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # A valid 3-coloring for the Fano plane
    # 0:0, 1:1, 2:2, 3:0, 4:1, 5:1, 6:0
    # Vertices 7, 8 are isolated, assign color 0
    colors = [0, 1, 2, 0, 1, 1, 0, 0, 0]
    
    return {
        'edges': edges,
        'colors': colors
    }

# EVOLVE-BLOCK-END
