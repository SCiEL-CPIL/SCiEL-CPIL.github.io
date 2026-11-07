"""Program for 'Few hyperedges forcing three colors'.

Constructs a 3-uniform hypergraph on vertices 0..8 that is exactly
3-chromatic (no proper 2-coloring exists) using the minimum possible
number of edges.  The optimal construction is the Fano plane (the unique
2-(7,3,1) design) embedded on vertices 0..6, which is the smallest known
3-chromatic 3-uniform hypergraph (7 edges).  Vertices 7 and 8 are isolated.

The result is verified at runtime:
  * every edge is a distinct 3-subset of {0..8};
  * the supplied 3-coloring leaves every edge non-monochromatic;
  * no proper 2-coloring exists (exhaustive check over all 2^9 assignments,
    the deterministic 'Min-SAT' infeasibility test).
"""

# EVOLVE-BLOCK-START
import itertools


def _fano_edges():
    # The seven lines of the Fano plane on vertices 0..6.
    return [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5),
    ]


def _fano_colors():
    # A proper 3-coloring of the Fano plane; vertices 7,8 are isolated.
    return [0, 1, 2, 0, 1, 2, 0, 0, 0]


def _edge_is_nonmonochromatic(edge, colors):
    c = {colors[v] for v in edge}
    return len(c) >= 2


def _has_proper_2coloring(edges):
    # Exhaustive deterministic check: does a 2-coloring exist in which no
    # edge is monochromatic?  (This is the 'Min-SAT' infeasibility test.)
    n = 9
    for bits in range(1 << n):
        ok = True
        for e in edges:
            # edge is monochromatic iff all three vertices share a color bit
            b0 = (bits >> e[0]) & 1
            b1 = (bits >> e[1]) & 1
            b2 = (bits >> e[2]) & 1
            if b0 == b1 == b2:
                ok = False
                break
        if ok:
            return True
    return False


def construct():
    edges = [list(e) for e in _fano_edges()]
    colors = list(_fano_colors())

    # --- runtime verification (deterministic, cheap) ---
    # distinct triples on 0..8
    assert len(set(map(tuple, edges))) == len(edges)
    for e in edges:
        assert len(set(e)) == 3
        assert all(isinstance(v, int) and 0 <= v <= 8 for v in e)
    # 3-coloring: nine labels from 0..2
    assert len(colors) == 9
    assert all(isinstance(c, int) and c in (0, 1, 2) for c in colors)
    # every edge non-monochromatic under the 3-coloring
    for e in edges:
        assert _edge_is_nonmonochromatic(e, colors)
    # exactly three-chromatic: no proper 2-coloring exists
    assert not _has_proper_2coloring(edges)

    return {'edges': edges, 'colors': colors}
# EVOLVE-BLOCK-END
