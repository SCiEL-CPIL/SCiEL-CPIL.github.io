"""
Rewritten program for the suite task.
Implements the theoretical minimum 3-uniform 3-chromatic hypergraph (Fano plane).
"""

# EVOLVE-BLOCK-START

def construct():
    """
    Returns a 3-uniform hypergraph on 9 vertices (0..8) with chromatic number 3
    and the minimum possible number of edges (7).
    
    The Fano plane is the unique 3-uniform hypergraph with 7 edges that is not
    2-colorable. It is defined on 7 vertices. We embed it in 9 vertices by
    leaving vertices 7 and 8 isolated.
    
    Returns:
        dict: A dictionary with keys 'edges' and 'colors'.
    """
    # The 7 edges of the Fano plane
    # Vertices 0-6 are used. 7 and 8 are isolated.
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # A valid 3-coloring for the Fano plane.
    # Colors for 0-6: 0, 1, 2, 1, 2, 0, 1
    # Colors for 7-8: 0, 0 (arbitrary, as they are isolated)
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Ensure types are strictly Python int
    edges = [[int(x) for x in e] for e in edges]
    colors = [int(c) for c in colors]
    
    return {
        'edges': edges,
        'colors': colors
    }

# EVOLVE-BLOCK-END
