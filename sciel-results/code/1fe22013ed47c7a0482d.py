"""Search small 3-regular high-girth graphs: generalized Petersen family + LUW fallback."""
import time
import math
import random
from collections import deque

# EVOLVE-BLOCK-START

def _moore_bound(k, g):
    """Lower bound on the number of vertices of a k-regular graph of girth g."""
    if g % 2 == 0:
        r = g // 2
        return 2 * k * (k ** (r - 1) - 1) // (k - 1)
    else:
        r = (g - 1) // 2
        return 1 + k * (k ** r - 1) // (k - 1)


def _girth_nb(adj):
    """Exact girth via non-backtracking BFS from every vertex (diameter <= girth-1)."""
    n = len(adj)
    best = n + 1  # > any real girth
    for s in range(n):
        # state (vertex, previous_vertex); prev = -1 means 'start'
        dist = {(-1, s): 0}
        q = deque([(-1, s)])
        while q:
            prev, v = q.popleft()
            d = dist[(prev, v)]
            if d >= best:
                continue
            for w in adj[v]:
                if w == prev:
                    continue
                st = (v, w)
                if st in dist:
                    if d + 1 < best:
                        best = d + 1
                else:
                    dist[st] = d + 1
                    q.append(st)
    return best if best <= n else n + 1


def _gp(n, k, perm):
    """Generalized Petersen graph GP(n,k) with permutation edge set (i, perm[i]) (mod n).
    Returns (n, edges) or None if it is not a simple 3-regular graph."""
    if n < 5:
        return None
    if k <= 0 or k >= n:
        return None
    if k == n // 2 and n % 2 == 0:
        return None
    edges = []
    seen = set()
    # outer cycle
    for i in range(n):
        j = (i + 1) % n
        a, b = (i, j) if i < j else (j, i)
        seen.add((a, b))
    # inner cycle via permutation
    for i in range(n):
        j = perm[i] % n
        if j == i:
            return None
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in seen:
            return None
        seen.add((a, b))
    # spokes
    for i in range(n):
        a, b = (i, i + n)
        seen.add((a, b))
    if len(seen) != 3 * n:
        return None
    for a, b in seen:
        edges.append((a, b))
    return n, edges


def _gp_girth(n, k, perm):
    """Girth of GP(n,k) with permutation edge set (i, perm[i])."""
    outer = list(range(n))
    inner = [x + n for x in perm]
    spokes = list(range(n))
    adj = [[] for _ in range(2 * n)]
    for i in range(n):
        a, b = outer[i], outer[(i + 1) % n]
        adj[a].append(b); adj[b].append(a)
        a, b = inner[i], inner[(i + 1) % n]
        adj[a].append(b); adj[b].append(a)
        a, b = spokes[i], spokes[i] + n
        adj[a].append(b); adj[b].append(a)
    for v in range(2 * n):
        if len(adj[v]) != 3:
            return 0
    return _girth_nb(adj)


def _luw(k, g):
    """Lazebnik-Ustimenko-Woldar CD(d, q) graph, q = k (prime or 4). Valid fallback."""
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
                y[c] = (add(x[c], mul(y[a], x[b])) if side == 0
                        else sub(x[c], mul(x[a], y[b])))
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
    deadline = t0 + 112.0  # safety margin under the 150 s kill
    rng = random.Random(12345)

    best = None  # (n, edges)

    def consider(n, edges):
        nonlocal best
        if best is None or n < best[0]:
            best = (n, edges)

    # ---- Generalized Petersen search (deterministic then randomized) ----
    nmax = 1200
    tried = set()

    def check(n, k, perm):
        nonlocal tried
        key = (n, k)
        if key in tried:
            return
        if time.time() > deadline - 3.0:
            return
        tried.add(key)
        if _gp(n, k, perm) is None:
            return
        gt = _gp_girth(n, k, perm)
        if gt >= g:
            consider(n, list(_gp(n, k, perm)[1]))

    # deterministic sweep over (n, k)
    for n in range(5, nmax + 1):
        if time.time() > deadline - 3.0:
            break
        if n % 2 == 0:
            kk = range(1, n // 2)
        else:
            kk = range(1, n)
        for k2 in kk:
            if (n, k2) in tried:
                continue
            check(n, k2, [(i * k2) % n for i in range(n)])

    # randomized permutation search on promising sizes
    if best is not None:
        cap = best[0] - 1
    else:
        cap = min(2 * nmax, 3 * _moore_bound(k, g))
    while time.time() < deadline - 4.0:
        n = rng.randint(5, max(5, cap))
        k2 = rng.randint(1, n - 1)
        if (n, k2) in tried:
            continue
        perm = [rng.randrange(n) for _ in range(n)]
        check(n, k2, perm)

    # ---- Fallback: guaranteed-valid LUW graph ----
    if best is None:
        if time.time() > deadline - 2.0:
            raise RuntimeError("out of time")
        consider(*_luw(k, g))

    return best[0], best[1]

# EVOLVE-BLOCK-END
