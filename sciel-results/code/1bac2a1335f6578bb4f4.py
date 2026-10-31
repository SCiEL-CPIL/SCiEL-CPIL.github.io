"""Combined program: LUW construction + voltage lifts + random 3-regular search."""
# EVOLVE-BLOCK-START
import time as _time
import random as _random
from collections import deque

# ---------------------------------------------------------------------------
# Fast girth verification using non-backtracking BFS from every vertex.
# State is (vertex, previous_vertex); a cycle shorter than g is found when we
# can reach the start vertex with the previous vertex being one of start's
# neighbours (i.e. we have walked back into start without backtracking).
# ---------------------------------------------------------------------------
def _verify_girth(adj, n, g):
    # girth >= g  <=>  no closed non-backtracking walk of length in [3, g-1]
    for start in range(n):
        dist = {}
        dq = deque()
        for w in adj[start]:
            dist[(start, w)] = 1
            dq.append((start, w))
        while dq:
            v, prev = dq.popleft()
            d = dist[(v, prev)]
            if d >= g - 1:
                continue
            nd = d + 1
            for w in adj[v]:
                if w == prev:
                    continue
                if w == start:
                    # closed non-backtracking walk of length nd
                    return False
                state = (w, v)
                if state not in dist:
                    dist[state] = nd
                    dq.append(state)
    return True


# ---------------------------------------------------------------------------
# LUW construction (algebraic, from the current program).
# ---------------------------------------------------------------------------
def _luw_build(k, g, q, d, rule):
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

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
    n = len(order)
    if n < 2 * k:
        return None
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    if any(dg != k for dg in deg):
        return None
    return n, edges


def _make_rule(d, variant=0):
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    if variant == 0:
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    elif variant == 1:
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(c0, 1), (1, c0 + 1), (c0 - 1, 1), (1, c0 - 2)][r]
    elif variant == 2:
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 + 1), (c0, 1), (c0 - 1, 1), (1, c0 - 2)][r]
    return rule


def _luw_search(k, g, deadline):
    best = None
    d_min = g - 5 if (g - 5) % 2 == 1 else g - 4
    d_min = max(3, d_min)
    q = k
    for d in range(d_min, d_min + 6, 2):
        if _time.time() > deadline:
            break
        for variant in range(3):
            if _time.time() > deadline:
                break
            rule = _make_rule(d, variant)
            valid = True
            for c in range(2, d + 1):
                a, b = rule[c]
                if a < 1 or a > d - 1 or b < 1 or b > d - 1 or a == b:
                    valid = False
                    break
            if not valid:
                continue
            result = _luw_build(k, g, q, d, rule)
            if result is not None:
                n, edges = result
                if best is None or n < best[0]:
                    adj = [[] for _ in range(n)]
                    for u, v in edges:
                        adj[u].append(v)
                        adj[v].append(u)
                    if _verify_girth(adj, n, g):
                        best = (n, edges)
    return best


# ---------------------------------------------------------------------------
# Voltage-graph lift of the Petersen graph over Z_m (3-regular).
# ---------------------------------------------------------------------------
_PETERSEN_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
    (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
    (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
]


def _build_lift(m, seed):
    rng = _random.Random(seed)
    n = 10 * m
    adj = [set() for _ in range(n)]
    edges = []
    for u, v in _PETERSEN_EDGES:
        vol = rng.randint(0, m - 1)
        for i in range(m):
            a = u * m + i
            b = v * m + (i + vol) % m
            if b not in adj[a]:
                adj[a].add(b)
                adj[b].add(a)
                edges.append((a, b))
    for i in range(n):
        if len(adj[i]) != 3:
            return None
    return n, [list(s) for s in adj], edges


# ---------------------------------------------------------------------------
# Random 3-regular graph via the configuration model (with a simple switch
# repair to avoid multi-edges / loops).
# ---------------------------------------------------------------------------
def _random_3reg(n, seed):
    rng = _random.Random(seed)
    stubs = []
    for v in range(n):
        stubs.extend([v, v, v])
    adj = [set() for _ in range(n)]
    edges = []
    for _ in range(20):
        rng.shuffle(stubs)
        adj = [set() for _ in range(n)]
        edges = []
        ok = True
        for i in range(0, n * 3, 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v or v in adj[u]:
                ok = False
                break
            adj[u].add(v)
            adj[v].add(u)
            edges.append((u, v))
        if not ok:
            continue
        # repair: remove any multi-edges / loops by swapping
        for _ in range(50):
            bad = []
            for u in range(n):
                for v in adj[u]:
                    if v == u or adj[v].count(u) > 1:
                        bad.append((u, v))
                        break
            if not bad:
                break
            u, v = bad[0]
            # find a swap partner
            fixed = False
            for w in range(n):
                for x in adj[w]:
                    if w == u or w == v or x == u or x == v:
                        continue
                    if u in adj[x] or v in adj[w]:
                        continue
                    if (u in adj[w] and v in adj[x]):
                        continue
                    adj[u].remove(v)
                    adj[v].remove(u)
                    adj[u].add(w)
                    adj[w].add(u)
                    adj[v].add(x)
                    adj[x].add(v)
                    fixed = True
                    break
                if fixed:
                    break
            if not fixed:
                break
        # final validity
        ok = True
        for u in range(n):
            if len(adj[u]) != 3:
                ok = False
                break
        if not ok:
            continue
        edges = []
        seen = set()
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
                    seen.add((u, v))
        if len(edges) != n * 3 // 2:
            continue
        # connectedness
        vis = {0}
        stack = [0]
        while stack:
            cur = stack.pop()
            for nb in adj[cur]:
                if nb not in vis:
                    vis.add(nb)
                    stack.append(nb)
        if len(vis) != n:
            continue
        return n, [list(s) for s in adj], edges
    return None


def construct(k, g):
    start = _time.time()
    deadline = start + 110  # safety margin before the 150 s kill

    # Moore bound (lower bound on n) for k-regular girth g
    if g % 2 == 0:
        h = g // 2
        moore = 2 * (k * (k - 1) ** (h - 1) - 1)
    else:
        h = (g - 1) // 2
        moore = 2 * (k * (k - 1) ** (h - 1) - 1) + k * (k - 1) ** (h - 1)
    moore = max(moore, 2 * k)

    # 1) Algebraic LUW construction (fast, guaranteed k-regular)
    best = _luw_search(k, g, deadline - 5)

    if k == 3 and _time.time() < deadline - 5:
        # 2) Voltage lifts of the Petersen graph
        m_min = max(1, (moore + 9) // 10)
        for m in range(m_min, m_min + 120):
            if _time.time() > deadline - 5:
                break
            n = 10 * m
            if n < moore:
                continue
            if best is not None and n >= best[0]:
                continue
            for attempt in range(15):
                if _time.time() > deadline - 3:
                    break
                res = _build_lift(m, attempt)
                if res is None:
                    continue
                n, adj, edges = res
                if _verify_girth(adj, n, g):
                    best = (n, edges)
                    break  # move to next (larger) m only if we want smaller; stop

        # 3) Random 3-regular graphs from the Moore bound upward
        if _time.time() < deadline - 10:
            n_try = moore
            if n_try % 2 != 0:
                n_try += 1
            while n_try < 4000 and _time.time() < deadline - 10:
                if best is not None and n_try >= best[0]:
                    break
                for attempt in range(25):
                    if _time.time() > deadline - 5:
                        break
                    res = _random_3reg(n_try, attempt)
                    if res is None:
                        continue
                    n, adj, edges = res
                    if _verify_girth(adj, n, g):
                        best = (n, edges)
                        break
                n_try += 8

    if best is None:
        # Fallback: LUW with the original rule
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        rule = _make_rule(d, 0)
        res = _luw_build(k, g, k, d, rule)
        if res is not None:
            best = res
        else:
            return 2 * k, [(i, i + k) for i in range(k)]

    return best
# EVOLVE-BLOCK-END
