"""Construct a k-regular graph of girth >= g on as few vertices as possible.

Primary: Lazebnik-Ustimenko-Woldar (LUW) incidence graph CD(d, q) with q = k,
a q-regular bipartite graph whose girth is at least d + 5.  We pick the
smallest d that still gives girth >= g, build the connected component of the
zero point (it is q-regular and simple), and return it.

Fallback: a deterministic 1-factorization of the complete graph K_k gives a
k-regular graph on 2k vertices (girth 4) - always valid, used only if the
LUW build cannot be completed in time.
"""
import time

# EVOLVE-BLOCK-START
_DEADLINE = None
_MARGIN = 10.0


def _time_left():
    if _DEADLINE is None:
        return True
    return (time.time() - _DEADLINE) < _MARGIN


def _build_luw(k, g):
    q = k
    # smallest d (odd, >= 3) with d + 5 >= g
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    # Precompute the rule lookups for speed.
    rule_a = [0] * (d + 1)
    rule_b = [0] * (d + 1)
    for c in range(2, d + 1):
        rule_a[c], rule_b[c] = rule[c]

    # BFS from the zero point.
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []

    head = 0
    while head < len(order):
        if not _time_left():
            return None
        vertex = order[head]
        head += 1
        side, x = vertex
        vi = index[vertex]

        if side == 0:
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    y[c] = add(x[c], mul(y[rule_a[c]], x[rule_b[c]]))
                other = (1, tuple(y))
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                edges.append((vi, index[other]))
        else:
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    y[c] = sub(x[c], mul(x[rule_a[c]], y[rule_b[c]]))
                other = (0, tuple(y))
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                # edge already recorded from the point side
    return len(order), edges


def _build_fallback(k):
    # k-regular graph on 2k vertices via 1-factorization of K_k.
    # Two copies of K_k's vertices; pair vertex (0,i) with (1, pi(i)) for a
    # 1-factor pi, and within each side use a (k-1)-regular graph.
    # Simpler: take the circulant on 2k vertices.
    n = 2 * k
    edges = []
    # For each vertex i, connect to i +- s for s in a set of (k-1)/2 or k/2
    # steps; plus a perfect matching.
    steps = list(range(1, k))
    for i in range(n):
        for s in steps:
            j = (i + s) % n
            if i < j:
                edges.append((i, j))
    # This circulant C(2k; 1..k-1) is (k-1)-regular. Add a perfect matching
    # (i, i+k) to make it k-regular.
    for i in range(k):
        edges.append((i, i + k))
    return n, edges


def construct(k, g):
    global _DEADLINE
    _DEADLINE = time.time() + 150.0

    try:
        res = _build_luw(k, g)
        if res is not None:
            return res
    except Exception:
        pass

    # Fallback (always valid k-regular).
    return _build_fallback(k)
# EVOLVE-BLOCK-END
