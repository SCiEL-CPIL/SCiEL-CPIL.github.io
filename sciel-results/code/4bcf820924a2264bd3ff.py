"""Deterministic optimal solution: Fano plane (7 edges) on vertices 0-6, isolated 7,8."""

def construct():
    # The Fano plane is the unique (up to isomorphism) 3-uniform hypergraph
    # on 7 vertices with chromatic number 3. It has exactly 7 edges.
    # Any 3-uniform hypergraph with fewer than 7 edges is 2-colorable,
    # so 7 is the minimum possible edge count for a 3-chromatic triple system.
    
    # Fano plane edges (standard representation):
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5],
    ]
    
    # A valid 3-coloring of the Fano plane:
    # Each edge must be non-monochromatic.
    # Let's verify:
    # v0=0, v1=1, v2=2, v3=1, v4=2, v5=0, v6=1
    # Edge [0,1,2]: colors 0,1,2 -> non-mono ✓
    # Edge [0,3,4]: colors 0,1,2 -> non-mono ✓
    # Edge [0,5,6]: colors 0,0,1 -> non-mono ✓
    # Edge [1,3,5]: colors 1,1,0 -> non-mono ✓
    # Edge [1,4,6]: colors 1,2,1 -> non-mono ✓
    # Edge [2,3,6]: colors 2,1,1 -> non-mono ✓
    # Edge [2,4,5]: colors 2,2,0 -> non-mono ✓
    
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    return {'edges': edges, 'colors': colors}
