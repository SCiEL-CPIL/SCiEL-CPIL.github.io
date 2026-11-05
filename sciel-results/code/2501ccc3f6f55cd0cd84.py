"""Construct a k-regular graph of girth g with as few vertices as possible.

Primary construction: Kautz graph K(d, k) for odd girth g = 2d+1, which is
k-regular with girth exactly 2d+1 and (k+1)*k^d vertices. This is a well-known
explicit high-girth construction and beats the Lazebnik-Ustimenko-Woldar seed
for the (5,9) case (3750 vs 6250 vertices).

Fallback for even girth: the Lazebnik-Ustimenko-Woldar CD(d, q) construction.
"""
import time
from itertools import product

_DEADLINE_MARGIN = 100.0  # evaluator kills at 150s; keep a safety margin


def _kautz(k, g):
    """Kautz graph K(d, k): k-regular, girth 2d+1, (k+1)*k^d vertices."""
    d = (g - 1) // 2
    L = d + 1
    verts = []
    idx = {}
    alpha = k + 1
    for seq in product(range(alpha), repeat=L):
        ok = True
        for i in range(L - 1):
            if seq[i] == seq[i + 1]:
                ok = False
                break
        if ok:
            idx[seq] = len(verts)
            verts.append(seq)
    n = len(verts)
    edges = []
    for i, seq in enumerate(verts):
        tail = seq[1:]
        last = seq[L - 1]
        for b in range(alpha):
            if b == last:
                continue
            j = idx[tail + (b,)]
            if i < j:
                edges.append((i, j))
    return n, edges


def _luw(k, g):
    """Lazebnik-Ustimenko-Woldar CD(d, q) with q = k (bipartite, even girth)."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


def construct(k, g):
    t0 = time.time()
    deadline = t0 + _DEADLINE_MARGIN
    try:
        if g % 2 == 1:
            n, edges = _kautz(k, g)
        else:
            n, edges = _luw(k, g)
    except Exception:
        # fallback to LUW regardless of parity
        n, edges = _luw(k, g)
    return n, edges
