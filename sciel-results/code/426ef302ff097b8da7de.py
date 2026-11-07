"""Rewritten program using the Fano plane (optimal 3-chromatic 3-uniform hypergraph)."""

def construct():
    # The Fano plane is the unique 3-uniform hypergraph on 7 vertices that is
    # 3-chromatic and edge-minimal. It has 7 edges.
    # Any 3-uniform hypergraph with fewer than 7 edges is 2-colorable.
    # We use vertices 0-6 for the Fano plane; vertices 7, 8 are isolated.
    
    edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    
    # Valid 3-coloring: each edge must be non-monochromatic.
    # Assignment: v0=0, v1=1, v2=2, v3=1, v4=2, v5=0, v6=1, v7=0, v8=0
    # Check each edge:
    # (0,1,2): 0,1,2 - non-mono ✓
    # (0,3,4): 0,1,2 - non-mono ✓
    # (0,5,6): 0,0,1 - non-mono ✓
    # (1,3,5): 1,1,0 - non-mono ✓
    # (1,4,6): 1,2,1 - non-mono ✓
    # (2,3,6): 2,1,1 - non-mono ✓
    # (2,4,5): 2,2,0 - non-mono ✓
    
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    return {'edges': list(edges), 'colors': list(colors)}
