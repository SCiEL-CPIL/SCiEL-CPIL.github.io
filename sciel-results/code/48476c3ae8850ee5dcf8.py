import time
import math
import random
from collections import deque

_START_TIME = time.time()
_TIMEOUT = 130


def _deadline_exceeded():
    return (time.time() - _START_TIME) > _TIMEOUT


def _verify_girth_fast(adj, n, g):
    """Return True iff no cycle of length < g. Level-by-level NB walk."""
    for start in range(n):
        frontier = [(start, nb) for nb in adj[start]]
        seen = set(frontier)
        level = 1
        while frontier and level < g:
            nxt = []
            for prev, curr in frontier:
                for nb2 in adj[curr]:
                    if nb2 == prev:
                        continue
                    st = (curr, nb2)
                    if st in seen:
                        if level + 1 < g:
                            return False
                    else:
                        seen.add(st)
                        nxt.append(st)
            frontier = nxt
            level += 1
    return True


def _check_simple_and_regular(edges, n, k):
    if len(edges) != n * k // 2:
        return False
    seen = set()
    deg = [0] * n
    for u, v in edges:
        if u == v:
            return False
        a, b = min(u, v), max(u, v)
        if (a, b) in seen:
            return False
        seen.add((a, b))
        deg[a] += 1
        deg[b] += 1
    for d in deg:
        if d != k:
            return False
    return True


def _valid(n, edges, k, g):
    if not _check_simple_and_regular(edges, n, k):
        return False
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return _verify_girth_fast(adj, n, g)


def _moore_bound(k, g):
    if g % 2 == 0:
        return 2 * sum((k - 1) ** i for i in range(g // 2))
    return 1 + k * sum((k - 1) ** i for i in range((g - 1) // 2))


def _is_prime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


# Lazebnik-Ustimenko-Woldar construction
def construct_lw(k, g):
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


# Alon-Babai-Suzuki construction
def construct_abs(k, g):
    mb = _moore_bound(k, g)
    q = 2
    while True:
        if _is_prime(q):
            if q ** 3 >= mb:
                break
        q += 1

    if k == 4:
        gens = [(1, 0, 0), (q - 1, 0, 0), (0, 1, 0), (0, q - 1, 1)]
    elif k == 6:
        gens = [(1, 0, 0), (q - 1, 0, 0), (0, 1, 0), (0, q - 1, 1), (0, 0, 1), (0, 0, q - 1)]
    else:
        return None

    n = q ** 3
    edges = []

    def idx(x, y, z):
        return (x * q + y) * q + z

    for x in range(q):
        for y in range(q):
            for z in range(q):
                u = idx(x, y, z)
                for gx, gy, gz in gens:
                    vx = (x + gx) % q
                    vy = (y + gy) % q
                    vz = (z + gz + x * gy) % q
                    v = idx(vx, vy, vz)
                    if u < v:
                        edges.append((u, v))
    return n, edges


# Circulant construction for even k
def construct_circulant(k, g, max_q=5000):
    if k % 2 != 0:
        return None
    half = k // 2
    mb = _moore_bound(k, g)
    start_q = mb if mb >= 2 * half + 1 else 2 * half + 1
    for q in range(start_q, start_q + max_q):
        if _deadline_exceeded():
            return None
        edges = []
        for u in range(q):
            for gi in range(1, half + 1):
                v = (u + gi) % q
                if u < v:
                    edges.append((u, v))
        if len(edges) != q * k // 2:
            continue
        adj = [[] for _ in range(q)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        if _verify_girth_fast(adj, q, g):
            return q, edges
    return None


# Random regular graph with local search for girth improvement
def _random_regular_graph(n, k, seed=42):
    """Generate a random k-regular graph on n vertices."""
    rng = random.Random(seed)
    # Use configuration model
    half_edges = n * k // 2
    stubs = []
    for i in range(n):
        stubs.extend([i] * k)
    rng.shuffle(stubs)
    
    adj = [set() for _ in range(n)]
    edges = []
    i = 0
    while i < len(stubs):
        u = stubs[i]
        j = i + 1
        while j < len(stubs):
            v = stubs[j]
            if v != u and v not in adj[u]:
                adj[u].add(v)
                adj[v].add(u)
                edges.append((u, v))
                stubs.pop(j)
                break
            j += 1
        else:
            # No valid partner found, retry with different shuffle
            return None
        i += 1
    
    if len(edges) != half_edges:
        return None
    return edges


def _local_improve_girth(n, edges, k, g, max_iters=5000):
    """Try to improve girth via edge swaps. Returns (n, edges, valid) or None."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))
    
    def is_valid():
        if not _check_simple_and_regular(edges, n, k):
            return False
        adj_list = [list(s) for s in adj]
        return _verify_girth_fast(adj_list, n, g)
    
    if is_valid():
        return n, edges, True
    
    rng = random.Random(12345)
    for it in range(max_iters):
        if _deadline_exceeded():
            break
        if len(edges) < 2:
            break
        i, j = rng.sample(range(len(edges)), 2)
        a, b = edges[i]
        c, d = edges[j]
        if len({a, b, c, d}) < 4:
            continue
        new1, new2 = (a, c), (b, d)
        n1, n2 = (min(new1), max(new1)), (min(new2), max(new2))
        if n1 in edge_set or n2 in edge_set or n1 == n2:
            continue
        # Apply
        edge_set.discard((min(a, b), max(a, b)))
        edge_set.discard((min(c, d), max(c, d)))
        edge_set.add(n1)
        edge_set.add(n2)
        edges[i], edges[j] = new1, new2
        adj[a].discard(b); adj[b].discard(a)
        adj[c].discard(d); adj[d].discard(c)
        adj[a].add(c); adj[c].add(a)
        adj[b].add(d); adj[d].add(b)
        
        if is_valid():
            return n, edges, True
        
        # Revert
        edge_set.discard(n1)
        edge_set.discard(n2)
        edge_set.add((min(a, b), max(a, b)))
        edge_set.add((min(c, d), max(c, d)))
        edges[i], edges[j] = (a, b), (c, d)
        adj[a].discard(c); adj[c].discard(a)
        adj[b].discard(d); adj[d].discard(b)
        adj[a].add(b); adj[b].add(a)
        adj[c].add(d); adj[d].add(c)
    
    return None


def construct(k, g):
    random.seed(12345)
    candidates = []
    
    def try_add(n, edges):
        if n is None or edges is None:
            return
        if len(edges) != n * k // 2:
            return
        candidates.append((n, edges))
    
    # Try LW construction (works for any k)
    lw = construct_lw(k, g)
    try_add(*lw)
    
    # Try ABS for even k
    if k % 2 == 0:
        res = construct_abs(k, g)
        try_add(*res)
    
    # Try circulant for even k
    if k % 2 == 0 and not _deadline_exceeded():
        res = construct_circulant(k, g)
        try_add(*res)
    
    # Sort by size and verify
    candidates.sort(key=lambda t: t[0])
    
    best = None
    for n, edges in candidates:
        if _deadline_exceeded():
            break
        if _check_simple_and_regular(edges, n, k):
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            if _verify_girth_fast(adj, n, g):
                best = (n, edges)
                break
            else:
                if not _deadline_exceeded():
                    res = _local_improve_girth(n, list(edges), k, g, max_iters=1000)
                    if res is not None:
                        best = (n, res[1])
                        break
    
    if best is not None:
        return best
    
    # Fallback: return LW
    return lw
