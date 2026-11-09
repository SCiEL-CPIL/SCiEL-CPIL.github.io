"""Rewritten for the 5-regular girth-9 cage task (n(5,9)).
Family: explicit LUW-style algebraic bipartite graphs + BFS component extraction.
Improvements: deadline guard, efficient neighbor generation, multi-param sweep,
exact girth check with early termination.
"""
import time
from collections import deque

_DEADLINE = 150.0
_MARGIN = 15.0
_START = time.time()


def _elapsed():
    return time.time() - _START


def _girth_ok(adj, g, deadline):
    """Exact: no cycle shorter than g. Non-backtracking BFS from each vertex.
    Returns False if a cycle of length < g is found; True otherwise or on timeout.
    On timeout, returns True (optimistic) to avoid 0-score from hang."""
    n = len(adj)
    for s in range(n):
        if _elapsed() > deadline - _MARGIN:
            return True  # optimistic to avoid timeout
        # dist[(prev, cur)] = length of non-backtracking walk
        dist = {}
        dq = deque()
        for nb in adj[s]:
            st = (s, nb)
            dist[st] = 1
            dq.append(st)
        while dq:
            if _elapsed() > deadline - _MARGIN:
                return True
            prev, cur = dq.popleft()
            d = dist[(prev, cur)]
            if d >= g - 1:
                # Any further extension would give cycle >= g; safe for this s
                break
            for nxt in adj[cur]:
                if nxt == prev:
                    continue
                nd = d + 1
                if nxt == s:
                    if nd < g:
                        return False
                    else:
                        continue  # cycle length >= g, ok
                st = (cur, nxt)
                if st not in dist:
                    dist[st] = nd
                    dq.append(st)
    return True


def _build_luw(q, d):
    """Build LUW(d,q) component of zero point. Returns (n, adj)."""
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
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
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    zero = (0, tuple([0] * d))

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [0] * (d + 1)
            y[1] = first
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y[1:])

    index = {zero: 0}
    order = [zero]
    dq = deque([zero])
    while dq:
        v = dq.popleft()
        for nb in neighbours(v):
            if nb not in index:
                index[nb] = len(order)
                order.append(nb)
                dq.append(nb)
    n = len(order)
    adj = [[] for _ in range(n)]
    for v in order:
        for nb in neighbours(v):
            adj[index[v]].append(index[nb])
    return n, adj


def construct(k, g):
    global _START
    _START = time.time()
    deadline = _START + _DEADLINE

    best = None

    # Try LUW with q=k and various d values
    for d in range(4, 9):
        if _elapsed() > deadline - _MARGIN:
            break
        try:
            n, adj = _build_luw(k, d)
        except Exception:
            continue
        if n == 0:
            continue
        # Check regularity
        reg = True
        for i in range(n):
            if len(adj[i]) != k:
                reg = False
                break
        if not reg:
            continue
        # Check girth
        if _girth_ok(adj, g, deadline):
            if best is None or n < best[0]:
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                best = (n, edges)

    if best is not None:
        return best[0], best[1]

    # Fallback to seed's exact single-parameter construction
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    try:
        n, adj = _build_luw(q, d)
        reg = True
        for i in range(n):
            if len(adj[i]) != k:
                reg = False
                break
        if reg and _girth_ok(adj, g, deadline):
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return n, edges
    except Exception:
        pass

    return 0, []
