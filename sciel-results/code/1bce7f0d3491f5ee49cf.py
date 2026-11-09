"""Construct a minimal 3-uniform 3-chromatic hypergraph on 9 vertices.

Returns the Fano plane (7 edges) on vertices 0-6, which is the theoretical
minimum for a 3-chromatic 3-uniform hypergraph. Vertices 7 and 8 are isolated.
"""

import time


def construct():
    t0 = time.time()
    limit = t0 + 85.0

    fano = (
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5),
    )

    palette = (0, 1, 2, 1, 2, 0, 1, 0, 0)

    for e in fano:
        if palette[e[0]] == palette[e[1]] == palette[e[2]]:
            raise ValueError("monochromatic edge in provided coloring")

    # Exhaustive 2-colorability check over 2^9 assignments.
    for mask in range(512):
        if time.time() > limit:
            raise RuntimeError("deadline exceeded")
        feasible = True
        for e in fano:
            a = (mask >> e[0]) & 1
            b = (mask >> e[1]) & 1
            c = (mask >> e[2]) & 1
            if a == b == c:
                feasible = False
                break
        if feasible:
            raise ValueError("found 2-coloring; logic error")

    out_edges = [tuple(int(x) for x in e) for e in fano]
    out_colors = [int(x) for x in palette]

    # Exact type / range checks
    assert isinstance(out_edges, list) and len(out_edges) == 7
    for e in out_edges:
        assert isinstance(e, tuple) and len(e) == 3
        for v in e:
            assert type(v) is int and 0 <= v <= 8
    assert isinstance(out_colors, list) and len(out_colors) == 9
    for c in out_colors:
        assert type(c) is int and c in (0, 1, 2)

    return {"edges": out_edges, "colors": out_colors}
