# EVOLVE-BLOCK-START

def construct():
    """
    Returns a 3-chromatic 3-uniform hypergraph on vertices 0..8 with minimal edges.
    Uses the Fano plane (7 edges) on vertices 0..6. Vertices 7 and 8 are isolated.
    """
    # Fano plane edges on vertices 0..6
    # These are the lines of the projective plane of order 2.
    # Each edge is a triple of distinct vertices.
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
    # Colors for vertices 0..8.
    # Vertices 7 and 8 are isolated, so their colors can be anything in {0,1,2}.
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Ensure types are strictly Python int and list
    # The literals above are already Python ints and lists, but we enforce it
    # to be absolutely safe against any environment quirks.
    edges = [[int(v) for v in e] for e in edges]
    colors = [int(c) for c in colors]
    
    # Lightweight validation to ensure correctness before returning
    # 1. Check edge count >= 1
    assert len(edges) >= 1
    
    # 2. Check edge structure: list of 3 distinct ints in 0..8
    for e in edges:
        assert isinstance(e, list)
        assert len(e) == 3
        assert all(isinstance(v, int) and not isinstance(v, bool) for v in e)
        assert all(0 <= v <= 8 for v in e)
        # Check distinct vertices in edge
        assert len(set(e)) == 3
        
    # 3. Check colors structure: 9 ints in 0..2
    assert len(colors) == 9
    assert all(isinstance(c, int) and not isinstance(c, bool) for c in colors)
    assert all(0 <= c <= 2 for c in colors)
    
    # 4. Check non-monochromatic edges
    for e in edges:
        a, b, c = e
        assert not (colors[a] == colors[b] == colors[c]), f"Edge {e} is monochromatic with colors {colors[a], colors[b], colors[c]}"
        
    # 5. Check distinct edges (no duplicate triples)
    edge_set = set(tuple(sorted(e)) for e in edges)
    assert len(edge_set) == len(edges), "Duplicate edges found"
    
    return {
        'edges': edges,
        'colors': colors
    }

# EVOLVE-BLOCK-END
