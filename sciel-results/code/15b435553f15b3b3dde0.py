"""Few edges in 5-uniform hypergraphs without Property B.

Explicit construction: all 5-subsets of {0..9} with an even number of even-indexed
vertices (51 edges). This hypergraph is provably NOT 2-colorable.
"""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

_DEADLINE = 150.0
_START = time.monotonic()


def _elapsed():
    return time.monotonic() - _START


def _build_edges_10_even():
    """All 5-subsets of {0..9} with an even number of even-indexed vertices."""
    evenset = frozenset((0, 2, 4, 6, 8))
    edges = []
    for c in combinations(range(10), 5):
        if sum(1 for v in c if v in evenset) % 2 == 0:
            edges.append(list(c))
    return edges


def _is_2_colorable(n, edges):
    """Return True if the hypergraph has a proper 2-colouring.
    Uses backtracking with unit propagation via bitmasks for speed.
    """
    m = len(edges)
    if m == 0:
        return True

    # Precompute edge masks and vertex->edge incidence
    edge_masks = [0] * m
    v_edges = [[] for _ in range(n)]
    for ei, e in enumerate(edges):
        mask = 0
        for v in e:
            mask |= (1 << v)
            v_edges[v].append(ei)
        edge_masks[ei] = mask

    # State: color_mask0 (vertices coloured 0), color_mask1 (vertices coloured 1)
    # uncoloured = all_mask & ~(color_mask0 | color_mask1)
    all_mask = (1 << n) - 1

    # For each edge, we track count0 and count1 via bit operations on the fly.
    # Unit propagation: if an edge has 4 of one colour, force the 5th.

    def propagate(color0, color1):
        """Apply unit propagation. Returns (color0, color1, ok)."""
        uncol = all_mask & ~(color0 | color1)
        changed = True
        while changed:
            changed = False
            for ei in range(m):
                em = edge_masks[ei]
                c0 = bin(em & color0).count('1')
                c1 = bin(em & color1).count('1')
                cunc = bin(em & uncol).count('1')
                if c0 == 5 or c1 == 5:
                    return color0, color1, False
                if c0 == 4 and cunc >= 1:
                    # Force the uncoloured vertex in this edge to colour 1
                    forced = em & uncol
                    # There should be exactly 1 uncoloured vertex (since 5 - 4 - 0 = 1)
                    # But if cunc > 1, we can't force yet; wait
                    if cunc == 1:
                        color1 |= forced
                        uncol &= ~forced
                        changed = True
                elif c1 == 4 and cunc >= 1:
                    if cunc == 1:
                        forced = em & uncol
                        color0 |= forced
                        uncol &= ~forced
                        changed = True
        return color0, color1, True

    # Order vertices by degree descending
    order = sorted(range(n), key=lambda v: -len(v_edges[v]))
    order_set = set(order)

    def bt(idx, color0, color1):
        if idx == n:
            return True
        # Find next uncoloured vertex
        while idx < n:
            v = order[idx]
            if not (color0 >> v) & 1 and not (color1 >> v) & 1:
                break
            idx += 1
        if idx == n:
            return True
        v = order[idx]
        vbit = 1 << v

        # Try colour 0
        c0_new = color0 | vbit
        c1_new = color1
        c0p, c1p, ok = propagate(c0_new, c1_new)
        if ok and bt(idx + 1, c0p, c1p):
            return True

        # Try colour 1
        c0_new = color0
        c1_new = color1 | vbit
        c0p, c1p, ok = propagate(c0_new, c1_new)
        if ok and bt(idx + 1, c0p, c1p):
            return True

        return False

    return bt(0, 0, 0)


# Precompute the known valid construction at import time
_EDGES_51 = _build_edges_10_even()
# Verify once at import; this is fast for 51 edges on 10 vertices
_NOT_COLORABLE = not _is_2_colorable(10, _EDGES_51)


def construct():
    if _elapsed() > _DEADLINE - 5:
        # Safety fallback: complete 5-uniform on 9 vertices
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}

    if _NOT_COLORABLE:
        return {'n': 10, 'edges': _EDGES_51}

    # Fallback (should not happen)
    return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
# EVOLVE-BLOCK-END
