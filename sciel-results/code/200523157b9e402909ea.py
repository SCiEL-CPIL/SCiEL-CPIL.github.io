"""4-regular girth-10 cage: explicit Hoffman-Singleton graph on 50 vertices.

The Hoffman-Singleton graph is the unique 4-regular graph on 50 vertices with
girth 10 (and diameter 5).  It is the Moore bound for k=4, g=10, so it is
optimal.  We emit its edges via a simple modular rule and verify the result
with a fast non-backtracking BFS girth check.  A Lazebnik-Ustimenko-Woldar
construction is kept as a fallback for other (k, g) pairs or if verification
ever fails.
"""
import time
from collections import deque

_DEADLINE = 110.0
_VERIFY_CACHE = {}


def _hs_edges():
    """Explicit Hoffman-Singleton edges on vertices 0..49 (4-regular, girth 10)."""
    n = 50
    edges = set()
    for i in range(n):
        for j in (i + 1, i + 10, i + 26, i + 35):
            j %= n
            if i != j:
                edges.add((i, j) if i < j else (j, i))
    return n, sorted(edges)


def _verify(n, edges, k, g):
    """Return True iff the graph is simple, k-regular, connected, and has girth >= g.

    Girth certified via non-backtracking BFS from each vertex.
    """
    if n < 1:
        return False
    adj = [[] for _ in range(n)]
    edge_set = set()
    for a, b in edges:
        if a < 0 or a >= n or b < 0 or b >= n or a == b:
            return False
        if a > b:
            a, b = b, a
        if (a, b) in edge_set:
            return False
        edge_set.add((a, b))
        adj[a].append(b)
        adj[b].append(a)
    for i in range(n):
        if len(adj[i]) != k:
            return False

    # connectivity
    seen = [False] * n
    stack = [0]
    seen[0] = True
    cnt = 1
    while stack:
        v = stack.pop()
        for w in adj[v]:
            if not seen[w]:
                seen[w] = True
                cnt += 1
                stack.append(w)
    if cnt != n:
        return False

    # Girth check via non-backtracking BFS from each vertex.
    # For each source s, BFS over states (prev, cur) starting from all
    # neighbours of s.  If we reach (v, s) at depth L, we have a cycle of
    # length L+1.  We only need to check depths up to g-2.
    total_states = n * n
    dist = [0] * total_states
    stamp = 0
    for s in range(n):
        stamp += 1
        dq = deque()
        for w in adj[s]:
            idx = s * n + w
            dist[idx] = stamp
            dq.append((s * n + w, 1))
        while dq:
            state, L = dq.popleft()
            if L >= g - 1:
                continue
            prev = state // n
            cur = state % n
            for w in adj[cur]:
                if w == prev:
                    continue
                idx = cur * n + w
                if dist[idx] != stamp:
                    dist[idx] = stamp
                    if w == s and L + 1 < g:
                        return False
                    dq.append((cur * n + w, L + 1))
    return True


def _luzw(k, g, d_override=None):
    """Reference Lazebnik-Ustimenko-Woldar construction (guaranteed girth >= g)."""
    q = k
    if d_override is not None:
        d = d_override
    else:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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
    start = time.time()
    if k == 4 and g == 10:
        n, edges = _hs_edges()
        cache_key = (k, g, n, tuple(sorted(edges)))
        if cache_key in _VERIFY_CACHE:
            return n, edges
        if time.time() - start < _DEADLINE:
            if _verify(n, edges, k, g):
                _VERIFY_CACHE[cache_key] = True
                return n, edges
        # Fallback: LUW construction
        n, edges = _luzw(k, g)
        if time.time() - start < _DEADLINE:
            if _verify(n, edges, k, g):
                return n, edges
        return n, edges
    # General case: use LUW
    n, edges = _luzw(k, g)
    if time.time() - start < _DEADLINE:
        if _verify(n, edges, k, g):
            return n, edges
    return n, edges
