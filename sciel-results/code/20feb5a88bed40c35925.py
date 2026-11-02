"""Improved LUW CD(d,k): pick the smallest d whose measured girth >= g.

The graph must be k-regular, so in the LUW CD(d,q) family q is fixed at k.
The seed overestimates d (for g=9 it uses d=5 -> n=2*k^5=6250).  Since n=2*k^d
grows with d, we scan d from 3 upward, measure the exact girth with a
non-backtracking BFS from every vertex, and return the first d with girth >= g.
This keeps the same construction family but yields a much smaller n (e.g. d=4,
girth 10 >= 9, n=2*k^4=1250 for (5,9)).  A deadline guard plus a seed-equivalent
fallback guarantee we never regress below the seed.
"""
import time
from collections import deque

_DEADLINE = None


def _build_luw(q, d):
    """Build the LUW graph CD(d, q). Returns (n, edges)."""
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
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
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
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


def _girth(n, edges, cap):
    """Exact girth via non-backtracking BFS from every vertex.

    Returns the shortest cycle length found, or cap+1 if no cycle shorter than
    cap+1 exists.  A non-backtracking closed walk from s of minimal length is a
    simple cycle through s, so the minimum over all s is the girth.
    """
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    best = cap + 1
    for s in range(n):
        dist = {}
        q = deque()
        for w in adj[s]:
            dist[(s, w)] = 1
            q.append((s, w))
        found = None
        while q:
            prev, v = q.popleft()
            d = dist[(prev, v)]
            if d >= best:
                continue
            for w in adj[v]:
                if w == prev:
                    continue
                if w == s:
                    cand = d + 1
                    if found is None or cand < found:
                        found = cand
                    continue
                key = (v, w)
                if key not in dist:
                    nd = d + 1
                    if nd < best:
                        dist[key] = nd
                        q.append((v, w))
        if found is not None and found < best:
            best = found
        if best == 3:
            break
    return best


def construct(k, g):
    global _DEADLINE
    start_time = time.time()
    _DEADLINE = start_time + 100.0  # safety margin well under the 120s limit

    q = k  # LUW CD(d,q) is q-regular, and the graph must be k-regular.
    # n = 2*q^d is strictly increasing in d, so the first d with girth >= g
    # is the minimum-n answer.
    for d in range(3, 80):
        if time.time() > _DEADLINE:
            break
        try:
            n, edges = _build_luw(q, d)
        except Exception:
            continue
        girth = _girth(n, edges, cap=g - 1)
        if girth >= g:
            return n, edges

    # Fallback: reproduce the seed's d choice so we never regress.
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    n, edges = _build_luw(q, d)
    return n, edges
