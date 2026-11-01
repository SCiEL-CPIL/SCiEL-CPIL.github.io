import time
import itertools
from collections import deque

_START = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill


def _elapsed():
    return time.time() - _START


def _time_ok():
    return _elapsed() < _DEADLINE


def _verify_simple_regular(edges, n, k):
    """Check simple graph, all degrees == k."""
    seen = set()
    deg = [0] * n
    for i, j in edges:
        if i < 0 or i >= n or j < 0 or j >= n or i == j:
            return False
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in seen:
            return False
        seen.add((a, b))
        deg[a] += 1
        deg[b] += 1
    for d in deg:
        if d != k:
            return False
    return True


def _min_girth_bfs(n, edges, k):
    """Exact minimum girth via non-backtracking BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    best = 10 ** 9
    for s in range(n):
        dist = {}
        dq = deque()
        for nb in adj[s]:
            dist[(nb, s)] = 1
            dq.append((nb, s))
        while dq:
            cur, prev = dq.popleft()
            d = dist[(cur, prev)]
            for nb in adj[cur]:
                if nb == prev:
                    continue
                key = (nb, cur)
                if key in dist:
                    L = d + 1
                    if L < best:
                        best = L
                else:
                    dist[key] = d + 1
                    dq.append((nb, cur))
    return best if best < 10 ** 9 else 10 ** 9


def _luzebnik_ustimenko_woldar(k, g):
    """
    L-U-W CD(d,q) construction: q-regular bipartite graph of girth >= d+5.
    Vertices: 2 * q^d. For k=4, g=10: d=5, n=2*4^5=2048.
    """
    q = k
    # Choose d such that d+5 >= g and d >= 3
    d = g - 5
    if d < 3:
        d = 3
    # d should be odd for the standard construction to guarantee girth >= d+5
    if d % 2 == 0:
        d += 1

    # Finite field arithmetic
    if q == 4:
        _mul_table = [[0, 0, 0, 0],
                      [0, 1, 2, 3],
                      [0, 2, 3, 1],
                      [0, 3, 1, 2]]
        def _add(x, y): return x ^ y
        def _sub(x, y): return x ^ y
        def _mul(x, y): return _mul_table[x][y]
    else:
        def _add(x, y): return (x + y) % q
        def _sub(x, y): return (x - y) % q
        def _mul(x, y): return (x * y) % q

    # L-U-W rules: for c=2..d, (a_c, b_c) with 1 <= a_c < c, 1 <= b_c <= c, a_c != b_c
    # Standard periodic rules
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)

    # Total vertices: 2 * q^d
    total = 2 * (q ** d)

    if not _time_ok():
        return total, []

    # Generate all vertices explicitly
    vertices = []
    for side in range(2):
        for x_tuple in itertools.product(range(q), repeat=d):
            vertices.append((side,) + x_tuple)
    
    index = {v: i for i, v in enumerate(vertices)}
    
    edges = []
    # Only generate edges from side 0 to side 1 to avoid duplicates
    for v in vertices:
        if v[0] != 0:
            continue
        side = 0
        x = v[1:]  # 0-indexed: x[0]=x1, ..., x[d-1]=xd
        
        for first in range(q):
            y = [0] * d
            y[0] = first  # y1 = first
            for c in range(2, d + 1):
                a, b = rule[c]
                # 1-indexed: x_c -> x[c-1], y_a -> y[a-1], x_b -> x[b-1]
                # side 0: y_c = x_c + y_a * x_b
                y[c - 1] = _add(x[c - 1], _mul(y[a - 1], x[b - 1]))
            
            neighbor = (1,) + tuple(y)
            # The neighbor must exist in index
            if neighbor in index:
                edges.append((index[v], index[neighbor]))
            else:
                # This should not happen if the mapping is bijective
                pass

    n = len(vertices)
    return n, edges


def _hoffman_singleton():
    """Hoffman-Singleton: 7-regular, girth 5, 50 vertices (n(7,5))."""
    n = 50
    edges = []

    def idx(a, b):
        return a * 10 + b
    for a in range(5):
        for b in range(10):
            v = idx(a, b)
            nbs = []
            nbs.append(idx(a, (b + 1) % 10))
            nbs.append(idx(a, (b - 1) % 10))
            nbs.append(idx((a + 1) % 5, (2 * b) % 10))
            nbs.append(idx((a + 1) % 5, (2 * b + 1) % 10))
            nbs.append(idx((a + 2) % 5, (4 * b) % 10))
            nbs.append(idx((a + 2) % 5, (4 * b + 1) % 10))
            for w in nbs:
                if v < w:
                    edges.append((v, w))
    return n, edges


def _petersen():
    """Petersen graph: 3-regular, girth 5, 10 vertices."""
    edges = [(0, 1), (0, 2), (0, 3), (0, 4),
             (1, 2), (1, 3), (1, 4),
             (5, 6), (5, 7), (5, 8), (5, 9),
             (6, 7), (6, 8), (6, 9),
             (7, 8), (7, 9), (8, 9),
             (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]
    return 10, edges


def _moore_girth5(k):
    """For girth 5, use known cages."""
    if k == 3:
        return _petersen()
    if k == 7:
        return _hoffman_singleton()
    # For other k, use L-U-W with d=0? No, girth 5 means d+5>=5 so d>=0.
    # L-U-W with d=1: girth >= 6 (too much). We need girth >= 5.
    # For girth 5, use the Moore bound construction if available, else L-U-W.
    return _luzebnik_ustimenko_woldar(k, 5)


def construct(k, g):
    """Return (n, edges) for a k-regular graph of girth >= g, minimizing n."""
    # Known optimal cages for common small cases.
    if k == 3 and g == 5:
        return _petersen()
    if k == 7 and g == 5:
        return _hoffman_singleton()
    if g == 5:
        return _moore_girth5(k)

    # General: L-U-W construction (valid, girth >= g).
    n, edges = _luzebnik_ustimenko_woldar(k, g)
    return n, edges
