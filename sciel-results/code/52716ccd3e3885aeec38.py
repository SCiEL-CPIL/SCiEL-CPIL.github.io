"""Model-and-optimize: verify a minimal 3-uniform 3-chromatic hypergraph on 9 vertices.

The minimum number of edges in a 3-uniform hypergraph that is 3-chromatic is 7
(the Fano plane). This program hard-codes the Fano plane and verifies:
  (a) every provided edge is non-monochromatic under the given coloring,
  (b) no 2-coloring of the 9 vertices makes every edge non-monochromatic.

The Fano plane is the unique (up to isomorphism) 7-edge 3-uniform 3-chromatic
hypergraph. Minimum edge count is 7.
"""

def construct():
    # Fano plane on vertices 0..6; vertices 7,8 are isolated
    edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]

    # A valid 3-coloring of the Fano plane
    # Vertices: 0:0, 1:1, 2:2, 3:1, 4:2, 5:0, 6:1, 7:0, 8:0
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]

    # --- Verification ---

    # (a) Check every edge is non-monochromatic under 'colors'
    for e in edges:
        c = {colors[v] for v in e}
        if len(c) < 2:
            raise ValueError(f"Edge {e} is monochromatic under provided colors")

    # (b) Check no 2-coloring exists (i.e., for every 2-coloring, at least one
    #     edge is monochromatic)
    #     Enumerate all 2^9 = 512 two-colorings of the 9 vertices.
    n = 9
    for mask in range(1 << n):
        # mask bit i -> color i (0 or 1)
        # We can optimize by pre-computing vertex colors, but 512 iterations is trivial.
        # c[i] = (mask >> i) & 1
        ok = True
        for e in edges:
            v0, v1, v2 = e
            c0 = (mask >> v0) & 1
            c1 = (mask >> v1) & 1
            c2 = (mask >> v2) & 1
            # Edge is monochromatic if all three are same
            if (c0 == c1) and (c1 == c2):
                ok = False
                break
        if ok:
            raise ValueError("Fano plane appears to be 2-colorable — logic error")

    # Return with exact types: edges as list of tuples, colors as list of ints
    # Ensure types are strictly Python int and not bool or numpy types
    result_edges = []
    for e in edges:
        result_edges.append((int(e[0]), int(e[1]), int(e[2])))
    
    result_colors = [int(c) for c in colors]

    result = {
        'edges': result_edges,
        'colors': result_colors
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
