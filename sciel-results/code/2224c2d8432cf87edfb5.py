"""Stronger construction for the 5-regular girth-9 cage problem."""
import time
import random
from collections import deque

_DEADLINE = 115.0  # safety margin vs 120s wall-clock


def _bipartite_luw(q, d):
    """Incidence graph of Lazebnik-Ustimenko-Woldar CD(d,q): q-regular bipartite,
    girth at least d+5 for odd d. Returns (n, edges)."""
    q = int(q)
    d = int(d)
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
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            if side == 0:
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b]))
            else:
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = sub(x[c], mul(x[a], y[b]))
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


def _girth_bfs(n, adj, limit):
    """Exact girth via non-backtracking BFS from every vertex. Returns None if no
    cycle of length < limit is found (i.e. girth >= limit)."""
    for s in range(n):
        dist = {s: 0}
        prev = {s: -1}
        dq = deque([s])
        while dq:
            u = dq.popleft()
            du = dist[u]
            if du + 1 >= limit:
                continue
            for v in adj[u]:
                if v == prev.get(u, -1):
                    continue
                if v not in dist:
                    dist[v] = du + 1
                    prev[v] = u
                    dq.append(v)
                else:
                    if du + 1 + dist[v] < limit:
                        return False
    return True


def construct(k, g):
    t0 = time.time()
    deadline = t0 + _DEADLINE

    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    n, edges = _bipartite_luw(q, d)

    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Verify the construction meets the contract (girth >= g).
    if not _girth_bfs(n, adj, g):
        # Should not happen for valid LUW; fall back to a safer d.
        n, edges = _bipartite_luw(q, d + 2)
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        if time.time() > deadline - 5:
            return n, edges

    # Bounded local search: try to shrink via 2-opt edge swaps that preserve
    # k-regularity and girth >= g. Each swap replaces two disjoint edges
    # (a,b),(c,d) with (a,c),(b,d) (or (a,d),(b,c)) if it stays simple,
    # keeps degree k, and keeps girth >= g. We only accept a swap if it
    # reduces some proxy for n (here: keep n, but prune isolated components
    # is not possible since graph is connected). Instead we try to find a
    # smaller valid graph by trying different (q,d) combos quickly.

    # Try a few alternative (q,d) pairs that yield smaller n, validating each.
    best = (n, edges)
    rng = random.Random(12345)

    # Candidate d values (odd, >= g-5) and q = k.
    candidates = []
    for dd in range(max(3, g - 5), max(3, g - 5) + 6, 2):
        if dd % 2 == 1:
            candidates.append(dd)
    candidates = sorted(set(candidates))

    for dd in candidates:
        if time.time() > deadline - 10:
            break
        try:
            n2, e2 = _bipartite_luw(q, dd)
        except Exception:
            continue
        if n2 < best[0]:
            adj2 = [[] for _ in range(n2)]
            for u, v in e2:
                adj2[u].append(v)
                adj2[v].append(u)
            if _girth_bfs(n2, adj2, g):
                best = (n2, e2)

    return best[0], best[1]
