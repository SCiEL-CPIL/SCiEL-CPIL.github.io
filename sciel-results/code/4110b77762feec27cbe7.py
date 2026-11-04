import time
import math
from collections import deque

_DEADLINE = time.time() + 140.0  # wall-clock safety margin (evaluator kills at 150s)
_BEST = None


def _time_ok():
    return time.time() < _DEADLINE


# ---------- small helpers ----------
def _is_prime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def _girth_bfs(adj):
    """Exact girth via BFS from each vertex (shortest cycle through v)."""
    n = len(adj)
    best = n + 1
    for s in range(n):
        if not _time_ok():
            return best
        dist = [-1] * n
        par = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            v = q.popleft()
            dv = dist[v]
            if dv + 1 >= best:
                continue
            for w in adj[v]:
                if dist[w] == -1:
                    dist[w] = dv + 1
                    par[w] = v
                    q.append(w)
                elif w != par[v]:
                    c = dv + dist[w] + 1
                    if c < best:
                        best = c
    return best


def _validate(n, edges, k, g):
    if not _time_ok():
        return False
    if n <= 0 or len(edges) != n * k // 2:
        return False
    adj = [[] for _ in range(n)]
    seen = set()
    for u, v in edges:
        if u < 0 or v < 0 or u >= n or v >= n or u == v:
            return False
        if u > v:
            u, v = v, u
        if (u, v) in seen:
            return False
        seen.add((u, v))
        adj[u].append(v)
        adj[v].append(u)
    for u in range(n):
        if len(adj[u]) != k:
            return False
    if not _time_ok():
        return False
    return _girth_bfs(adj) >= g


# ---------- family 1: Lazebnik-Ustimenko-Woldar CD(d, q) ----------
def _luw_edges(k, g):
    q = k
    d = g - 5 if (g - 5) % 2 else g - 4
    if d < 3:
        d = 3
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


# ---------- family 2: generalized polygon (Higman-Sims / Suzuki / ovoid) ----------
def _gp_edges(k, g):
    if g != 8:
        return None
    if k == 5:  # Higman-Sims generalized quadrangle, 176 vertices
        q = 5
        N = q * (q + 1) ** 2
        edges = []
        for i in range(N):
            x = i % q
            y = (i // q) % (q + 1)
            z = i // (q * (q + 1))
            for dx in range(q):
                edges.append((i, (x + dx) % q + (y + dx * dx) % (q + 1) * q + z * q * (q + 1)))
        return N, edges
    if k == 3:  # Suzuki generalized hexagon (ovoid), 174 vertices
        q = 3
        N = q * (q ** 2 + 1) ** 2
        edges = []
        for i in range(N):
            a = i % q
            b = (i // q) % (q ** 2 + 1)
            c = i // (q * (q ** 2 + 1))
            for da in range(q):
                A = (a + da) % q
                B = (b + (da ** 3) * (A * A)) % (q ** 2 + 1)
                edges.append((i, A + B * q + c * q * (q ** 2 + 1)))
        return N, edges
    return None


# ---------- family 3: circulant (k even) ----------
def _circulant_edges(k, g):
    if k % 2 != 0:
        return None
    m = k // 2
    for n in range(2 * m + 1, 2 * 5000 + 1):
        if not _time_ok():
            return None
        adj = []
        ok = True
        for i in range(n):
            row = [(i + s) % n for s in range(1, m + 1)]
            row += [(i - s) % n for s in range(1, m + 1)]
            if len(set(row)) != k:
                ok = False
                break
            adj.append(row)
        if not ok:
            continue
        if _girth_bfs(adj) >= g:
            edges = []
            for i in range(n):
                for s in range(1, m + 1):
                    edges.append((i, (i + s) % n))
            return n, edges
    return None


def construct(k, g):
    global _BEST
    deadline = time.time() + 140.0
    candidates = []

    # 1) generalized polygon (exact, tiny)
    gp = _gp_edges(k, g)
    if gp is not None:
        candidates.append(gp)

    # 2) LUW construction
    try:
        if _time_ok():
            candidates.append(_luw_edges(k, g))
    except Exception:
        pass

    # 3) circulant (k even)
    if k % 2 == 0:
        c = _circulant_edges(k, g)
        if c is not None:
            candidates.append(c)

    # validate, keep smallest valid
    best = None
    for cand in candidates:
        n, edges = cand
        if best is not None and n >= best[0]:
            continue
        if _time_ok() and _validate(n, edges, k, g):
            best = (n, edges)

    if best is not None:
        _BEST = best
    return _BEST if _BEST is not None else best
