"""Search for a small 3-regular graph of girth >= g, falling back to LUW construction."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k; returns (n, edges)."""
    q = k
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


def _girth_at_least(adj, n, g):
    """Return True iff graph (adjacency list) has girth >= g.
    Exact: for even g, every radius (g//2 - 1) BFS ball must be a tree."""
    if g <= 2:
        return True
    radius = g // 2 - 1  # for g=16 -> 7
    # expected number of vertices in a tree ball of radius `radius` in k-regular
    # 1 + k*(2^radius - 1)
    k = len(adj[0])
    expected = 1 + k * ((1 << radius) - 1)
    for s in range(n):
        # BFS with parent tracking; detect any edge that closes a cycle within radius
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        stack = [s]
        count = 1
        for v in stack:
            if dist[v] >= radius:
                continue
            nd = dist[v] + 1
            for u in adj[v]:
                if dist[u] == -1:
                    dist[u] = nd
                    parent[u] = v
                    count += 1
                    if nd <= radius:
                        stack.append(u)
                elif u != parent[v] and dist[u] <= nd:
                    # an edge within the ball that is not a tree edge -> cycle
                    # cycle length = dist[v] + dist[u] + 1; must be >= g
                    if dist[v] + dist[u] + 1 < g:
                        return False
        if count != expected:
            return False
    return True


def _random_regular_graph(n, k, rng):
    """Generate a random simple k-regular graph on n vertices (n*k even).
    Returns adjacency list or None on failure."""
    if n * k % 2 != 0 or n < k + 1:
        return None
    half = list(range(n * k // 2))
    # configuration model
    stubs = []
    for v in range(n):
        stubs.extend([v] * k)
    adj = [set() for _ in range(n)]
    for _attempt in range(200):
        rng.shuffle(stubs)
        adj = [set() for _ in range(n)]
        ok = True
        for i in range(0, n * k, 2):
            a, b = stubs[i], stubs[i + 1]
            if a == b or b in adj[a]:
                ok = False
                break
            adj[a].add(b)
            adj[b].add(a)
        if ok and all(len(x) == k for x in adj):
            return [sorted(x) for x in adj]
    return None


def _deadline_reached(t0, limit):
    return (time.time() - t0) > limit


def construct(k, g):
    t0 = time.time()
    HARD_LIMIT = 140.0  # safety margin under 150s wall-clock

    # Moore bound lower bound on n
    if g >= 4:
        moore = 2 * (1 << (g // 2 - 1))  # for k=3: 2*(2^r -1)+2 ... use safe lower
        # For k=3, g even: n >= 2*(2^(g/2-1) - 1) + 2? Use simple: 2*(2^(g/2 -1))
    else:
        moore = 0

    # Start search from a reasonable lower bound up to a cap.
    # The LUW graph gives an upper bound; use it to cap the search.
    rng = random.Random(12345)

    # Build a list of candidate n values: even numbers (n*k must be even -> n even for k odd)
    # k=3 -> n must be even.
    step = 2 if k % 2 == 1 else 1
    lo = max(moore, k + 1)
    if lo % step:
        lo += step
    hi = lo + 400  # search a window above the Moore bound

    best = None
    n = lo
    attempts_per_n = 0
    # Try increasing n; for each, attempt a few random regular graphs.
    while n <= hi and not _deadline_reached(t0, HARD_LIMIT):
        attempts_per_n = 0
        max_attempts = 40 if n < 600 else 15
        while attempts_per_n < max_attempts and not _deadline_reached(t0, HARD_LIMIT):
            adj = _random_regular_graph(n, k, rng)
            attempts_per_n += 1
            if adj is None:
                continue
            if _girth_at_least(adj, n, g):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
        n += step

    # Fallback: guaranteed-valid LUW construction
    try:
        return _luw_construct(k, g)
    except Exception:
        # last resort: a trivial valid k-regular graph (only if g small); else raise
        raise
# EVOLVE-BLOCK-END
