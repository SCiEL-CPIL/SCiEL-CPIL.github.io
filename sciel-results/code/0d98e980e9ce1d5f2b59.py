import time
import numpy as np

def _xorshift64(state):
    state ^= (state << 13) & 0xFFFFFFFFFFFFFFFF
    state ^= state >> 7
    state ^= (state << 17) & 0xFFFFFFFFFFFFFFFF
    return state & 0xFFFFFFFFFFFFFFFF

def _randrange(rng, n):
    return _xorshift64(rng) % n

def _pairing_regular(n, k, rng):
    """Return adjacency (list of lists) of a random simple k-regular graph on n
    vertices via a random perfect matching of n*k stubs. Returns None on
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
        u = int(stubs[perm[a]])
        v = int(stubs[perm[a + 1]])
        if u == v:
            return None
        if v in seen[u]:
            return None
        adj[u].append(v)
        adj[v].append(u)
        seen[u].add(v)
        seen[v].add(u)
    return adj

def _girth_ok(adj, g):
    """Return True iff the graph has no cycle of length < g.
    Uses BFS on non-backtracking states (u, prev) from each source s, with an
    early cutoff at depth g-1."""
    n = len(adj)
    adj_t = [tuple(a) for a in adj]
    
    for s in range(n):
        dist = {}
        queue = []
        for v in adj_t[s]:
            key = v * n + s
            dist[key] = 1
            queue.append((v, s))
        head = 0
        while head < len(queue):
            u, p = queue[head]
            head += 1
            d = dist[u * n + p]
            if d >= g - 1:
                continue
            for w in adj_t[u]:
                if w == p:
                    continue
                if w == s:
                    return False
                key = w * n + u
                if key not in dist:
                    dist[key] = d + 1
                    queue.append((w, u))
    return True

def _luw_fallback(k, g):
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

    rule_a = [0] * (d + 1)
    rule_b = [0] * (d + 1)
    for c in range(2, d + 1):
        rule_a[c], rule_b[c] = rule[c]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            if side == 0:
                for c in range(2, d + 1):
                    a, b = rule_a[c], rule_b[c]
                    y[c] = add(x[c], mul(y[a], x[b]))
            else:
                for c in range(2, d + 1):
                    a, b = rule_a[c], rule_b[c]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        vidx = index[vertex]
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((vidx, index[other]))
    return len(order), edges

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second margin before 120s limit
    
    # Moore bound (lower bound on n)
    r = (g - 1) // 2
    moore = 1
    for i in range(r):
        moore += k * (k - 1) ** i
    n_start = moore
    if n_start < 2 * k:
        n_start = 2 * k
    
    # Search increasing n; budget trials by n so small n gets more effort.
    rng = 0x9E3779B97F4A7C15
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

    # Guaranteed fallback: the Lazebnik-Ustimenko-Woldar construction
    return _luw_fallback(k, g)
