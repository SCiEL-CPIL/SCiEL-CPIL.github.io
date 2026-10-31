"""RCS (randomized constructive search) for 3-regular girth-16 cages, with a
fast non-backtracking girth verifier and a LUW algebraic fallback."""
import time
import random
from collections import deque

_DEADLINE = time.time() + 142.0   # safety margin under the 150 s wall-clock kill


def _time_left():
    return _DEADLINE - time.time()


def _girth_ok(adj, g):
    """Exact girth check: no non-backtracking closed walk shorter than g.

    Uses the classic (u,v) non-backtracking BFS (Bosak). O(n * g).
    Returns True iff every non-backtracking cycle has length >= g.
    """
    n = len(adj)
    # state (u,v): last step into u coming from v
    dist = [[-1] * n for _ in range(n)]
    dq = deque()
    for u in range(n):
        for v in adj[u]:
            dist[v][u] = 0
            dq.append((v, u))
    while dq:
        u, v = dq.popleft()
        duv = dist[u][v]
        if duv + 1 >= g:
            continue
        for w in adj[u]:
            if w == v:
                continue
            if dist[w][u] == -1:
                dist[w][u] = duv + 1
                dq.append((w, u))
            elif dist[w][u] == duv + 1:
                # non-backtracking cycle of length duv+1+duv+1 <= g-1
                return False
    return True


def _random_girth16(k=3, g=16, seed=0, time_budget=120.0):
    """Randomized constructive search for a k-regular girth-g graph.

    Returns (n, edges) on success, else None.
    """
    rng = random.Random(seed)
    # girth-16 cubic: Moore bound >= 2*(2^7 - 1) = 254, best known 936.
    for n in range(254, 941, 2):
        if n % 2 != 0:
            continue
        if _time_left() < 3.0:
            break
        if n % 200 == 0:
            print("RCS trying n =", n)
        # --- build a random n-vertex 3-regular multigraph (perfect 3-matching)
        for attempt in range(10):
            if _time_left() < 3.0:
                break
            deg = [3] * n
            pool = [i for i in range(n) for _ in range(3)]
            rng.shuffle(pool)
            adj = [[] for _ in range(n)]
            ok = True
            i = 0
            while i < len(pool):
                a = pool[i]
                i += 1
                while i < len(pool) and (pool[i] == a or deg[a] == 0 or deg[pool[i]] == 0):
                    i += 1
                if i >= len(pool):
                    ok = False
                    break
                b = pool[i]
                i += 1
                adj[a].append(b)
                adj[b].append(a)
                deg[a] -= 1
                deg[b] -= 1
            if not ok:
                continue
            # --- simple (no multi-edges)
            if any(len(set(nb)) != 3 for nb in adj):
                continue
            # --- girth >= 16
            if _girth_ok(adj, g):
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges
    return None


def _luw(k=3, g=16):
    """Lazebnik-Ustimenko-Woldar CD(d,q) construction, connected component of 0."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
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


def construct(k, g):
    global _DEADLINE
    _DEADLINE = time.time() + 142.0

    # 1) try randomized constructive search for a small girth-g k-regular graph
    if k == 3 and g == 16:
        best = None
        for seed in range(0, 400):
            if _time_left() < 4.0:
                break
            res = _random_girth16(3, 16, seed=seed)
            if res is not None:
                n = res[0]
                if best is None or n < best[0]:
                    best = res
                if n <= 600:          # comfortably better than LUW; stop early
                    break
        if best is not None:
            return best

    # 2) fallback: guaranteed-valid LUW construction
    return _luw(k, g)
