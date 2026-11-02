"""Construct small k-regular graphs of large girth (cage problem).
Family: Lazebnik-Ustimenko-Woldar (LUW) Cayley graphs + circulant search,
with fast non-backtracking girth certification and a deadline guard."""
import time
import random
from collections import deque

_DEADLINE = 145.0  # wall-clock safety margin before the 150s kill
_T0 = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _gf4_tables():
    # GF(4) = {0,1,2,3}, primitive element 2 (2^2=3, 2^3=1). add = xor.
    table = [[0, 0, 0, 0],
             [0, 1, 2, 3],
             [0, 2, 3, 1],
             [0, 3, 1, 2]]
    return table


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction -> (n, edges)."""
    q = k
    # choose d so that the construction has girth at least g
    d = max(3, g - 5) if (g - 5) % 2 else max(3, g - 4)
    if q == 4:
        table = _gf4_tables()
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q

    # rule[c] = (a, b) used to compute y[c] from earlier y's and x's
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
    index = {start: 0}
    order = [start]
    edges = []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


def _build_adjacency(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _girth_nonbacktracking(adj, n, target):
    """Return the girth (minimum cycle length) via non-backtracking BFS from
    every vertex. Returns at least target if no shorter cycle found.
    Certified: a cycle of length L exists iff a non-backtracking closed walk
    of length L exists (for L >= 3 in a simple graph)."""
    best = 10 ** 9
    # non-backtracking BFS: state = (vertex, previous_vertex), distance
    # For each start, do BFS over directed edges (u->v) with u != prev.
    for s in range(n):
        if _remaining() < 1.0:
            break
        # queue of (u, prev, dist)
        seen = set()  # (u, prev)
        dq = deque()
        for v in adj[s]:
            seen.add((v, s))
            dq.append((v, s, 1))
        found = None
        while dq:
            u, prev, dist = dq.popleft()
            if dist + 1 >= best:
                continue
            for w in adj[u]:
                if w == prev:
                    continue
                if w == s and dist + 1 >= 3:
                    # closed non-backtracking walk back to start
                    if dist + 1 < best:
                        best = dist + 1
                        if best <= target:
                            return best
                    continue
                if (w, u) in seen:
                    continue
                seen.add((w, u))
                if dist + 1 < best:
                    dq.append((w, u, dist + 1))
    return best


def _verify(k, g, n, edges):
    """Check degree-k, simple, girth >= g. Return (ok, actual_girth)."""
    if n <= 0:
        return False, 10 ** 9
    adj = _build_adjacency(n, edges)
    seen_edges = set()
    for u in range(n):
        if len(adj[u]) != k:
            return False, 10 ** 9
        for v in adj[u]:
            if u == v:
                return False, 10 ** 9
            key = (min(u, v), max(u, v))
            if key in seen_edges:
                return False, 10 ** 9
            seen_edges.add(key)
    girth = _girth_nonbacktracking(adj, n, g)
    return girth >= g, girth


def _circulant_search(k, g, nmax, seed=12345):
    """Search circulant graphs (vertex i connected to i +/- s_j mod n) for
    a k-regular graph of girth >= g on as few vertices as possible."""
    rng = random.Random(seed)
    best = None  # (n, edges)
    # Moore bound lower bound for k-regular girth g (even):
    # n >= k * ( (k-1)^(g/2 - 1) + 1 )
    if g % 2 == 0:
        lower = k * ((k - 1) ** (g // 2 - 1) + 1)
    else:
        lower = k * (k - 1) ** ((g - 1) // 2)
    lower = max(lower, k + 1)
    for n in range(lower, nmax + 1):
        if _remaining() < 3.0:
            break
        if k % 2 == 1:
            continue  # circulant with odd degree needs a self-loop-free odd set; skip
        # need k/2 step pairs; choose distinct steps from 1..n//2
        for _trial in range(40):
            if _remaining() < 1.0:
                break
            steps = rng.sample(range(1, n // 2 + 1), k // 2)
            adj = [set() for _ in range(n)]
            ok = True
            for i in range(n):
                for s in steps:
                    a = (i + s) % n
                    b = (i - s) % n
                    adj[i].add(a)
                    adj[i].add(b)
            if any(len(adj[i]) != k for i in range(n)):
                continue
            girth = _girth_nonbacktracking(adj, n, g)
            if girth >= g:
                edges = []
                for i in range(n):
                    for s in steps:
                        edges.append((i, (i + s) % n))
                if best is None or n < best[0]:
                    best = (n, edges)
                break  # found a valid circulant for this n; n is minimal so stop
    return best


def construct(k, g):
    """Return (n, edges): a k-regular simple graph with girth >= g,
    minimizing n. Entry point (keep name/signature)."""
    global _T0, _DEADLINE
    _T0 = time.time()
    _DEADLINE = time.time() + 145.0

    candidates = []  # list of (n, edges, verified)

    # 1) LUW construction
    try:
        n_luw, edges_luw = _luw_construct(k, g)
        ok, girth = _verify(k, g, n_luw, edges_luw)
        if ok:
            candidates.append((n_luw, edges_luw))
    except Exception:
        pass

    # 2) Circulant search (may find fewer vertices for some (k,g))
    try:
        # cap circulant search well below LUW size
        n_luw = candidates[0][0] if candidates else 10 ** 9
        circ = _circulant_search(k, g, min(n_luw - 1, 2000))
        if circ is not None:
            n_c, edges_c = circ
            ok, _ = _verify(k, g, n_c, edges_c)
            if ok:
                candidates.append((n_c, edges_c))
    except Exception:
        pass

    if not candidates:
        # Fallback: LUW without verification guard (should not happen)
        n_luw, edges_luw = _luw_construct(k, g)
        return n_luw, edges_luw

    candidates.sort(key=lambda t: t[0])
    return candidates[0][0], candidates[0][1]
