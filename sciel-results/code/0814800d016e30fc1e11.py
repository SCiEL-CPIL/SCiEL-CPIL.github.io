"""
Minimal 3-chromatic 3-uniform hypergraph on 9 vertices.

The Fano plane (7 edges on 7 vertices) is the proven minimum 3-chromatic
3-uniform hypergraph. Vertices 7 and 8 are isolated.
"""

import time

def construct():
    # The Fano plane: 7 edges on vertices 0..6
    edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5),
    ]

    # A valid 3-coloring: every edge is non-monochromatic
    # Vertex colors: 0:0, 1:1, 2:2, 3:1, 4:2, 5:0, 6:1, 7:0, 8:0
    # Check:
    #   (0,1,2) -> (0,1,2) OK
    #   (0,3,4) -> (0,1,2) OK
    #   (0,5,6) -> (0,0,1) OK (not all same)
    #   (1,3,5) -> (1,1,0) OK
    #   (1,4,6) -> (1,2,1) OK
    #   (2,3,6) -> (2,1,1) OK
    #   (2,4,5) -> (2,2,0) OK
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]

    # Verify: every edge is non-monochromatic under 'colors'
    for e in edges:
        c0, c1, c2 = colors[e[0]], colors[e[1]], colors[e[2]]
        if c0 == c1 == c2:
            raise ValueError(f"Edge {e} is monochromatic under provided colors")

    # Verify: no 2-coloring makes all edges non-monochromatic.
    # Enumerate all 2^9 = 512 assignments.
    # An edge is monochromatic under a 2-coloring if all three vertices have the same color.
    # We need: for EVERY 2-coloring, at least one edge is monochromatic.
    n = 9
    # Pre-extract edge vertex indices for speed
    ev = [(e[0], e[1], e[2]) for e in edges]
    
    for mask in range(1 << n):
        ok = True
        for a, b, c in ev:
            # Get colors for vertices a, b, c
            ca = (mask >> a) & 1
            cb = (mask >> b) & 1
            cc = (mask >> c) & 1
            if ca == cb == cc:
                # This edge is monochromatic, so this 2-coloring fails
                ok = False
                break
        if ok:
            # Found a 2-coloring where no edge is monochromatic -> hypergraph is 2-colorable
            raise ValueError("Hypergraph is 2-colorable — logic error")

    # Return with exact types
    result = {
        'edges': list(edges),
        'colors': list(colors),
    }

    # Final type assertions
    assert isinstance(result['edges'], list)
    assert len(result['edges']) >= 1
    for e in result['edges']:
        assert isinstance(e, tuple)
        for v in e:
            assert isinstance(v, int) and not isinstance(v, bool)
            assert 0 <= v <= 8
    assert isinstance(result['colors'], list)
    assert len(result['colors']) == 9
    for c in result['colors']:
        assert isinstance(c, int) and not isinstance(c, bool)
        assert c in (0, 1, 2)

    return result
