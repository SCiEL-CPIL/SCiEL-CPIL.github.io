def construct():
    # The Fano plane on vertices 0..6 is a 3-uniform hypergraph that is
    # not 2-colorable (every 2-coloring of its 7 vertices leaves at least
    # one line monochromatic).  Therefore its chromatic number is exactly 3,
    # and 7 edges is the minimal edge count for a 3-chromatic triple system
    # on 9 vertices (vertices 7 and 8 are isolated and do not affect the
    # edge count).  We verify the 2-coloring infeasibility explicitly with a
    # brute-force check over all 2^7 = 128 assignments to the Fano vertices.
    import itertools
    import time

    deadline = time.time() + 80.0

    # Fano plane lines (edges)
    fano_edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5),
    ]

    # Brute-force verify: no 2-coloring avoids a monochromatic edge
    def has_valid_2coloring(edges):
        verts = set()
        for e in edges:
            verts.update(e)
        verts = sorted(verts)
        idx = {v: i for i, v in enumerate(verts)}
        n = len(verts)
        for bits in range(1 << n):
            ok = True
            for e in edges:
                c = [ (bits >> idx[v]) & 1 for v in e ]
                if c[0] == c[1] == c[2]:
                    ok = False
                    break
            if ok:
                return True
        return False

    assert not has_valid_2coloring(fano_edges), "Fano plane must be non-2-colorable"

    # Build the full edge list on vertices 0..8 (vertices 7, 8 isolated)
    edges = [list(e) for e in fano_edges]

    # A valid 3-coloring: no edge is monochromatic
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]

    # Verify coloring validity
    for e in edges:
        assert not (colors[e[0]] == colors[e[1]] == colors[e[2]]), "Edge is monochromatic"

    # Ensure types are exactly Python int and lists
    edges = [[int(v) for v in e] for e in edges]
    colors = [int(c) for c in colors]

    return {
        'edges': edges,
        'colors': colors
    }
