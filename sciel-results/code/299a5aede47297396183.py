"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Returns a 3-uniform hypergraph on 9 vertices with 7 edges (Fano plane + 2 isolated vertices)
    which is not 2-colorable, and a valid 3-coloring.
    
    The Fano plane is the smallest 3-uniform hypergraph with chromatic number 3.
    It has 7 vertices and 7 edges. We embed it in vertices 0-6 and leave 7-8 isolated.
    
    Edges (Fano Plane):
    0: {0, 1, 2}
    1: {0, 3, 4}
    2: {0, 5, 6}
    3: {1, 3, 5}
    4: {1, 4, 6}
    5: {2, 3, 6}
    6: {2, 4, 5}
    
    Valid 3-Coloring:
    v0: 0
    v1: 1
    v2: 2
    v3: 1
    v4: 1
    v5: 2
    v6: 2
    v7: 0 (isolated)
    v8: 0 (isolated)
    
    Check edges:
    {0,1,2}: 0,1,2 (OK)
    {0,3,4}: 0,1,1 (OK)
    {0,5,6}: 0,2,2 (OK)
    {1,3,5}: 1,1,2 (OK)
    {1,4,6}: 1,1,2 (OK)
    {2,3,6}: 2,1,2 (OK)
    {2,4,5}: 2,1,2 (OK)
    """
    
    # Fano plane edges on vertices 0-6
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # Colors for vertices 0-8
    # Using the verified 3-coloring above
    colors = [0, 1, 2, 1, 1, 2, 2, 0, 0]
    
    # Ensure strict Python types as per contract
    # Convert lists to lists of ints, colors to list of ints
    edges = [[int(v) for v in e] for e in edges]
    colors = [int(c) for c in colors]
    
    return {
        'edges': edges,
        'colors': colors
    }
# EVOLVE-BLOCK-END
