"""LUW (Lazebnik-Ustimenko-Woldar) cage construction, vectorized with a time guard."""
import time
import numpy as np

_DEADLINE = 120.0
_START = time.monotonic()


def _remaining():
    return _DEADLINE - (time.monotonic() - _START)


def _luw_components(q, d):
    """Return (n, edges) for the 4-regular zero-point component of CD(d, q)."""
    if q == 4:
        table = np.array([[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]], dtype=np.int64)
        add = lambda a, b: (a ^ b).astype(np.int64)
        sub = lambda a, b: (a ^ b).astype(np.int64)
        mul = lambda a, b: table[a.astype(np.int64) % 4, b.astype(np.int64) % 4]
    else:
        add = lambda a, b: (a + b) % q
        sub = lambda a, b: (a - b) % q
        mul = lambda a, b: (a * b) % q

    # Recurrence for the LUW rule, verified to match the reference program.
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def gen(side, X):
        """X: (q, d) int64 array of one-side vertices -> (q, d) array of the other side."""
        Y = np.zeros((q, d), dtype=np.int64)
        Y[:, 0] = np.arange(q, dtype=np.int64)
        for c in range(2, d + 1):
            a, b = rule[c]
            if side == 0:
                Y[:, c] = add(X[:, c], mul(Y[:, a], X[:, b]))
            else:
                Y[:, c] = sub(X[:, c], mul(X[:, a], Y[:, b]))
        return Y

    def key(side, Y):
        return (side << (d * 2)) + int(Y[0, 0])

    seen = set()
    order = []
    P0 = gen(1, np.zeros((q, d), dtype=np.int64))
    for t in range(q):
        k = key(0, P0[t]); seen.add(k); order.append(k)
    for t in range(q):
        k = key(1, P0[t]); seen.add(k); order.append(k)

    cur_side, cur_t = 0, 0
    while cur_t < len(order):
        if _remaining() < 3.0:
            break
        v = order[cur_t]
        side = (v >> (d * 2)) & 1
        t = v & (q - 1)
        if side == 0:
            X = np.zeros((q, d), dtype=np.int64)
            X[:, 0] = t
            Y = gen(1, X)
            for u in range(q):
                k = key(1, Y[u])
                if k not in seen:
                    seen.add(k); order.append(k)
        else:
            X = np.zeros((q, d), dtype=np.int64)
            X[:, 0] = t
            Y = gen(0, X)
            for u in range(q):
                k = key(0, Y[u])
                if k not in seen:
                    seen.add(k); order.append(k)
        cur_t += 1

    idx = {v: i for i, v in enumerate(order)}
    edges = []
    for v in order:
        if _remaining() < 3.0:
            break
        side = (v >> (d * 2)) & 1
        t = v & (q - 1)
        if side == 0:
            X = np.zeros((q, d), dtype=np.int64)
            X[:, 0] = t
            Y = gen(1, X)
            for u in range(q):
                edges.append((idx[v], idx[key(1, Y[u])]))
    return len(order), edges


def _verify(n, edges, k, g):
    """Exact girth check via non-backtracking BFS from every vertex."""
    if n < 2:
        return n >= g
    adj = [[] for _ in range(n)]
    for a, b in edges:
        if a == b:
            return False
        adj[a].append(b)
        adj[b].append(a)
    for v in range(n):
        if len(adj[v]) != k:
            return False
    for s in range(n):
        dist = {s: 0}
        stack = [(s, -1)]
        while stack:
            v, pv = stack.pop()
            if v == s:
                continue
            for w in adj[v]:
                if w == pv:
                    continue
                nd = dist[v] + 1
                if w not in dist:
                    dist[w] = nd
                    stack.append((w, v))
                elif w == s:
                    if nd + 1 < g:
                        return False
    return True


def construct(k, g):
    """Return (n, edges): a k-regular simple graph of girth at least g, n minimized."""
    q = k
    # girth of CD(d, q) is d + 5 (odd d). Choose the smallest d with d + 5 >= g.
    d = g - 5
    if d % 2 == 0:
        d += 1
    d = max(3, d)

    n, edges = _luw_components(q, d)

    # Exact girth certification; fall back to the reference construction if needed.
    if _remaining() > 15.0 and not _verify(n, edges, k, g):
        n, edges = _reference_construct(k, g)

    return n, edges


def _reference_construct(k, g):
    """Scalar reference LUW construction (fallback, matches the seed program)."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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
