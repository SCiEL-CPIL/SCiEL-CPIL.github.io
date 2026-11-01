"""Combined program: configuration-model + simulated-annealing local search,
with LUW / voltage-lift fallbacks for guaranteed valid output."""
# EVOLVE-BLOCK-START
import time as _time
import random as _random
import math
from collections import deque

try:
    import numpy as _np
    _HAVE_NP = True
except Exception:
    _np = None
    _HAVE_NP = False


# --------------------------- girth detection ---------------------------
def _girth_shortest(adj, n, g, start, limit):
    """Return length of the shortest non-backtracking cycle through `start`
    (length >= 3), or limit if none shorter than `limit` is found."""
    if _HAVE_NP:
        dist = _np.full((n, n), limit, dtype=_np.int32)
        dist[-1, start] = 0
        dq = deque()
        dq.append((start, -1, 0))
        while dq:
            v, prev, d = dq.popleft()
            if d >= limit - 1:
                continue
            for w in adj[v]:
                if w == prev:
                    continue
                nd = d + 1
                if w == start:
                    if nd >= 3:
                        return nd
                    continue
                if dist[v, w] > nd:
                    dist[v, w] = nd
                    dq.append((w, v, nd))
        return limit
    else:
        dist = {}
        dist[(start, -1)] = 0
        dq = deque()
        dq.append((start, -1, 0))
        while dq:
            v, prev, d = dq.popleft()
            if d >= limit - 1:
                continue
            for w in adj[v]:
                if w == prev:
                    continue
                nd = d + 1
                if w == start:
                    if nd >= 3:
                        return nd
                    continue
                key = (w, v)
                if key not in dist or dist[key] > nd:
                    dist[key] = nd
                    dq.append((w, v, nd))
        return limit


def _verify_girth(adj, n, g):
    """True iff every vertex has no non-backtracking cycle shorter than g."""
    for start in range(n):
        if _girth_shortest(adj, n, g, start, g - 1) < g:
            return False
    return True


def _girth_of(adj, n, g):
    """Return the actual girth (min over vertices), capped at g."""
    best = g
    for start in range(n):
        v = _girth_shortest(adj, n, g, start, g)
        if v < best:
            best = v
        if best <= 3:
            break
    return best


# --------------------------- LUW construction ---------------------------
def _luw_construct(q, g):
    p = q
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    add, sub, mul = (lambda x, y: (x + y) % p), (lambda x, y: (x - y) % p), (lambda x, y: x * y % p)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(p):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
        yield 0, ()  # placeholder never used

    # Build graph via BFS over state space
    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        side, x = vertex
        for first in range(p):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            other = 1 - side, tuple(y)
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if side == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


# --------------------------- configuration model ---------------------------
def _config_model_3reg(n, rng, max_tries=40):
    if n < 4 or n % 2 != 0:
        return None
    for _ in range(max_tries):
        stubs = []
        for v in range(n):
            stubs += [v, v, v]
        rng.shuffle(stubs)
        adj = [[] for _ in range(n)]
        ok = True
        for i in range(0, n * 3, 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v or v in adj[u]:
                ok = False
                break
            adj[u].append(v)
            adj[v].append(u)
        if not ok:
            continue
        seen = [False] * n
        stack = [0]
        seen[0] = True
        cnt = 1
        while stack:
            x = stack.pop()
            for y in adj[x]:
                if not seen[y]:
                    seen[y] = True
                    cnt += 1
                    stack.append(y)
        if cnt == n:
            return adj
    return None


def _edges_from_adj(adj):
    n = len(adj)
    seen = set()
    edges = []
    for u in range(n):
        for v in adj[u]:
            key = (u, v) if u < v else (v, u)
            if key not in seen:
                seen.add(key)
                edges.append((u, v))
    return edges


# --------------------------- simulated-annealing local search ---------------------------
def _anneal_3reg(n, g, seed, time_budget):
    rng = _random.Random(seed)
    adj = _config_model_3reg(n, rng)
    if adj is None:
        return None

    start_t = _time.time()
    T = 2.0
    cool = 0.9995

    for step in range(600000):
        if _time.time() - start_t > time_budget:
            break
        if step % 200 == 0:
            T *= cool

        # pick random edge (a,b)
        a = rng.randrange(n)
        b = adj[a][rng.randrange(3)]
        # pick another disjoint random edge (c,d)
        chosen = False
        for _try in range(10):
            c = rng.randrange(n)
            d = adj[c][rng.randrange(3)]
            if c == a or c == b or d == a or d == b:
                continue
            if d not in adj[c]:
                continue
            if c in adj[a] or d in adj[b]:
                continue
            chosen = True
            break
        if not chosen:
            continue

        g_before = _girth_of(adj, n, g)
        if g_before >= g:
            return n, _edges_from_adj(adj)

        # apply swap (a,b)+(c,d) -> (a,c)+(b,d)
        adj[a].remove(b); adj[b].remove(a)
        adj[c].remove(d); adj[d].remove(c)
        adj[a].append(c); adj[c].append(a)
        adj[b].append(d); adj[d].append(b)

        g_after = _girth_of(adj, n, g)
        if g_after >= g:
            return n, _edges_from_adj(adj)

        if g_after < g_before:
            # worsening: revert with high probability
            if rng.random() > math.exp(-(g_before - g_after) / max(T, 1e-6)):
                adj[a].remove(c); adj[c].remove(a)
                adj[b].remove(d); adj[d].remove(b)
                adj[a].append(b); adj[b].append(a)
                adj[c].append(d); adj[d].append(c)

    if _verify_girth(adj, n, g):
        return n, _edges_from_adj(adj)
    return None


# --------------------------- voltage-lift fallback ---------------------------
def _get_base_graphs():
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    k33_edges = [
        (0, 3), (0, 4), (0, 5),
        (1, 3), (1, 4), (1, 5),
        (2, 3), (2, 4), (2, 5)
    ]
    return [(petersen_edges, 10), (k33_edges, 6)]


def _build_lift(base_edges, n_base, m, voltages):
    n = n_base * m
    adj = [set() for _ in range(n)]
    edges = []
    for u, v in base_edges:
        vol = voltages[(u, v)]
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
    adj_list = [sorted(adj[i]) for i in range(n)]
    return n, adj_list, edges


def _voltage_lift_search(base_edges, n_base, m, g, seed, n_attempts=15):
    rng = _random.Random(seed)
    for _ in range(n_attempts):
        voltages = {}
        for u, v in base_edges:
            vol = rng.randint(0, m - 1)
            voltages[(u, v)] = vol
            voltages[(v, u)] = (-vol) % m
        result = _build_lift(base_edges, n_base, m, voltages)
        if result is None:
            continue
        n, adj_list, edges = result
        if _verify_girth(adj_list, n, g):
            return n, edges
    return None


# --------------------------- main entry ---------------------------
def construct(k, g):
    start_time = _time.time()
    deadline = start_time + 145

    if k != 3:
        return _luw_construct(k, g)

    if g % 2 == 0:
        moore = 2 * (2 ** (g // 2) - 1)
    else:
        moore = 2 * (2 ** (g // 2 - 1) - 1) + 2

    best_result = None
    best_n = float('inf')

    # Primary: configuration-model + SA local search
    n_start = max(moore, 6)
    if n_start % 2 != 0:
        n_start += 1
    n_end = max(n_start + 64, min(n_start + 400, 1400))
    n_try = n_start
    while n_try <= n_end and _time.time() < deadline - 30:
        if n_try >= best_n:
            break
        if n_try % 2 != 0:
            n_try += 1
            continue
        res = _anneal_3reg(n_try, g, seed=n_try * 104729 + 7, time_budget=18.0)
        if res is not None:
            n_r, edges_r = res
            if n_r < best_n:
                best_n = n_r
                best_result = (n_r, edges_r)
            break
        n_try += 2

    # Fallback: voltage lifts
    if _time.time() < deadline - 20:
        for base_edges, n_base in _get_base_graphs():
            if _time.time() > deadline - 15:
                break
            m_min = max(1, (moore + n_base - 1) // n_base)
            for m in range(m_min, m_min + 400):
                if _time.time() > deadline - 12:
                    break
                n_lift = n_base * m
                if n_lift < moore or n_lift >= best_n:
                    continue
                res = _voltage_lift_search(base_edges, n_base, m, g,
                                           seed=m * 100000 + n_base, n_attempts=15)
                if res is not None:
                    n_r, edges_r = res
                    if n_r < best_n:
                        best_n = n_r
                        best_result = (n_r, edges_r)
                    break

    # Guaranteed fallback: LUW
    if best_result is None or _time.time() < deadline - 5:
        n_luw, edges_luw = _luw_construct(3, g)
        if best_result is None or n_luw < best_n:
            best_result = (n_luw, edges_luw)

    return best_result
# EVOLVE-BLOCK-END
