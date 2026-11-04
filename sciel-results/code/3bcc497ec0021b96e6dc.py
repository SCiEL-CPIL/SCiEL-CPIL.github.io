"""Search for a small 5-regular graph with girth >= g.

Strategy: generate random k-regular graphs via stub matching, verify girth
exactly and fast, keep the smallest valid one. A deadline guard returns the
best valid result before the wall-clock limit.

Key speedups vs the seed:
  * Girth verification uses a per-source BFS on non-backtracking states with a
    flat distance array and an early cutoff at depth g-1, so it is O(n * m)
    total instead of O(n^2 * m) with a 2-D Python matrix per source.
  * The search tries increasing n with a time-budgeted trial count, so small
    n (the interesting regime) gets the most effort.
"""
import time
import numpy as np

# ---------------- deterministic RNG ----------------
def _xorshift64(state):
    state ^= (state << 13) & 0xFFFFFFFFFFFFFFFF
    state ^= state >> 7
    state ^= (state << 17) & 0xFFFFFFFFFFFFFFFF
    return state & 0xFFFFFFFFFFFFFFFF

def _randrange(rng, n):
    return _xorshift64(rng) % n

# ---------------- k-regular graph generation ----------------
def _pairing_regular(n, k, rng):
    """Return adjacency (list of lists) of a random simple k-regular graph on n
    vertices via a random perfect matching of n*k stubs.  Returns None on
    failure (self-loop or parallel edge)."""
    m = n * k
    stubs = np.repeat(np.arange(n, dtype=np.int64), k)
    perm = np.arange(m, dtype=np.int64)
    for i in range(m - 1, 0, -1):
        j = _randrange(rng, i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    adj = [[] for _ in range(n)]
    seen = [set() for _ in range(n)]
    for a in range(0, m, 2):
        u = stubs[perm[a]]
        v = stubs[perm[a + 1]]
        if u == v:
            return None
        if v in seen[u]:
            return None
        adj[u].append(v)
        adj[v].append(u)
        seen[u].add(v)
        seen[v].add(u)
    return adj

# ---------------- exact girth via non-backtracking walks ----------------
def _girth_ok(adj, g):
    """Return True iff the graph has no cycle of length < g.
    Uses BFS on non-backtracking states (u, prev) from each source s, with an
    early cutoff at depth g-1.  A cycle of length L < g is found when a
    non-backtracking walk of length L-1 from a neighbour of s returns to s."""
    n = len(adj)
    for s in range(n):
        # dist indexed by (prev, u) -> depth; use a flat dict keyed by prev*n+u
        # but we only go to depth g-1, so a small dict is fine and fast.
        dist = {}
        queue = []
        for v in adj[s]:
            dist[v * n + s] = 1
            queue.append((v, s))
        head = 0
        while head < len(queue):
            u, p = queue[head]
            head += 1
            d = dist[u * n + p]
            if d >= g - 1:
                continue
            for w in adj[u]:
                if w == p:
                    continue
                if w == s:
                    return False
                key = w * n + u
                if key not in dist:
                    dist[key] = d + 1
                    queue.append((w, u))
    return True

# ---------------- main entry point ----------------
def construct(k, g):
    deadline = time.time() + 115.0   # safety margin under the 150 s limit
    rng = 0x9E3779B97F4A7C15
    best = None
    best_n = None

    def consider(n, adj):
        nonlocal best, best_n
        if _girth_ok(adj, g):
            best = adj
            best_n = n
            return True
        return False

    # Moore bound (lower bound on n)
    r = (g - 1) // 2
    moore = 1
    for i in range(r):
        moore += k * (k - 1) ** i
    n_start = moore
    if n_start < 2 * k:
        n_start = 2 * k

    # Search increasing n; budget trials by n so small n gets more effort.
    n = n_start
    while n <= 1200 and time.time() < deadline:
        if n < 400:
            trials = 400
        elif n < 700:
            trials = 120
        elif n < 1000:
            trials = 40
        else:
            trials = 15

        for _ in range(trials):
            if time.time() > deadline:
                break
            adj = _pairing_regular(n, k, rng)
            if adj is None:
                continue
            if _girth_ok(adj, g):
                return n, [(i, j) for i in range(n) for j in adj[i] if i < j]

        n += 1

    # (3) guaranteed fallback: the Lazebnik-Ustimenko-Woldar construction
    if best is not None:
        return best_n, [(i, j) for i in range(best_n) for j in best[i] if i < j]

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
