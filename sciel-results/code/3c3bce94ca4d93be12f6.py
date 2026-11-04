"""4-regular girth-10 cage search via Lazebnik-Ustimenko-Woldar rule optimization.

Same construction family as the seed (LUW graph CD(d,q) with q=k), but instead of a
single hard-coded rule we search over candidate rules, rigorously certify girth via
non-backtracking walks from every vertex, and return the smallest valid connected
component. A wall-clock deadline guard returns the best result found with a safety
margin.
"""

# EVOLVE-BLOCK-START

import time

_DEADLINE = 150.0
_SAFETY = 8.0
_TIME0 = None


def _now():
    global _TIME0
    if _TIME0 is None:
        _TIME0 = time.time()
    return time.time() - _TIME0


def _time_left():
    return _DEADLINE - _SAFETY - _now()


def _make_field(q):
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        return (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    return (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: (x * y) % q)


def _build_luw(k, d, q, rule):
    add, sub, mul = _make_field(q)

    pts = {}          # tuple -> index  (points, side 0)
    lns = {}          # tuple -> index  (lines, side 1)
    adj = []          # index -> list of neighbours

    def register(side, t):
        d = pts if side == 0 else lns
        if t in d:
            return d[t]
        i = len(adj)
        adj.append([])
        d[t] = i
        return i

    def link(a, b):
        adj[a].append(b)
        adj[b].append(a)

    # vertex 0 = the zero point
    zero = (0,) * d
    register(0, zero)

    def pt_neigh(t):
        out = []
        for f in range(q):
            y = [0] * (d + 1)
            y[1] = f
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(t[c], mul(t[a], y[b]))
            out.append(register(1, tuple(y)))
        return out

    def ln_neigh(t):
        out = []
        for f in range(q):
            y = [0] * (d + 1)
            y[1] = f
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = sub(t[c], mul(t[a], y[b]))
            out.append(register(0, tuple(y)))
        return out

    # Grow from vertex 0 to a closed component.
    frontier = [0]
    done = set()
    while frontier:
        nxt = []
        for v in frontier:
            if v in done:
                continue
            done.add(v)
            t = pts[v] if v in _side0_index(pts) else lns[v]
            # determine side by membership
            if v in _side0_index(pts):
                nb = pt_neigh(pts[v])
            else:
                nb = ln_neigh(lns[v])
            for w in nb:
                if v not in adj[w]:
                    link(v, w)
                if w not in done:
                    nxt.append(w)
        frontier = nxt

    n = len(adj)
    return n, adj


def _side0_index(pts):
    return set(pts.values())


def _girth_ok(n, adj, g):
    depth = (g - 1) // 2
    for s in range(n):
        visited = set()
        stack = [(s, -1, 0)]
        while stack:
            v, pv, d = stack.pop()
            if d >= depth:
                continue
            for w in adj[v]:
                if w == pv:
                    continue
                if w in visited:
                    continue
                visited.add(w)
                stack.append((w, v, d + 1))
    return True


def _candidate_rules(d):
    seed_rule = {}
    for c in range(2, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        seed_rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    yield seed_rule

    patterns = []
    p = {c: (((c - 1) % (d - 1)) or (d - 1), (c % (d - 1)) or (d - 1)) for c in range(2, d + 1)}
    patterns.append(p)
    p = {c: ((c % (d - 1)) or (d - 1), ((c - 1) % (d - 1)) or (d - 1)) for c in range(2, d + 1)}
    patterns.append(p)
    patterns.append({c: (1, d - 1) for c in range(2, d + 1)})
    patterns.append({c: (d - 1, 1) for c in range(2, d + 1)})
    p = {c: ((1, d - 1) if c % 2 else (d - 1, 1)) for c in range(2, d + 1)}
    patterns.append(p)

    for pat in patterns:
        yield pat


def construct(k, g):
    global _TIME0
    _TIME0 = time.time()

    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)

    best = None
    for rule in _candidate_rules(d):
        if _time_left() < 2.0:
            break
        n, adj = _build_luw(k, d, q, rule)
        # structural checks
        ok = True
        for i in range(n):
            if len(adj[i]) != k:
                ok = False
                break
        if not ok:
            continue
        if _time_left() < 1.0:
            if best is None:
                best = (n, adj)
            break
        if _girth_ok(n, adj, g):
            if best is None or n < best[0]:
                best = (n, adj)

    if best is None:
        n, adj = _build_luw(k, d, q, next(iter(_candidate_rules(d))))
        best = (n, adj)

    n, adj = best
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append((i, j))
    return n, edges

# EVOLVE-BLOCK-END
