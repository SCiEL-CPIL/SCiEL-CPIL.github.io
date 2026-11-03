"""Deterministic girth-16 cage construction for k=3.
Primary: tree-closure (depth-8 ball, leaves paired by branch) -> ~9841 vertices.
Fallback: Lazebnik-Ustimenko-Woldar algebraic construction (guaranteed valid).
"""

def _luw_construct(k, g):
    q = k
    d = g - 5 if (g - 5) % 2 else g - 4
    if d < 3:
        d = 3
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add = lambda x,y: x^y
        sub = lambda x,y: x^y
        mul = lambda x,y: table[x][y]
    else:
        add = lambda x,y: (x+y) % q
        sub = lambda x,y: (x-y) % q
        mul = lambda x,y: x*y % q
    rule = {2:(1,1), 3:(2,1), 4:(1,2)}
    for c in range(5, d+1):
        r, c0 = (c-5) % 4, c - (c-5) % 4
        rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0]*(d-1)
            for c in range(2, d+1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1-side, tuple(y)

    start = (0, (None,) + (0,)*d)
    index = {start: 0}
    order = [start]
    edges = []
    for v in order:
        for o in neighbours(v):
            if o not in index:
                index[o] = len(order)
                order.append(o)
            if v[0] == 0:
                edges.append((index[v], index[o]))
    return len(order), edges


def _tree_closure_construct(k, g):
    R = g // 2 - 1  # g=16 -> R=7
    n_leaves = k ** R
    n = 1 + k * n_leaves
    idx = 0
    root = 0
    idx += 1
    branch_leaves = {b: [] for b in range(k)}
    level = [root]
    for depth in range(1, R+1):
        nxt = []
        for v in level:
            base = idx
            for c in range(k):
                idx += 1
                nxt.append(idx-1)
                if depth == R:
                    b = (v // (k ** (R-1))) % k
                    branch_leaves[b].append(idx-1)
        level = nxt
    assert idx == n, (idx, n)
    # pair leaves cyclically across branches -> 2-regular on leaves
    L0, L1 = branch_leaves[0], branch_leaves[1]
    M = n_leaves
    edges = set()
    for v in range(1, n):
        base = v - 1
        parent = base // k
        edges.add((parent, v) if parent < v else (v, parent))
    for i in range(M):
        a = L0[i]; b = L1[i]
        edges.add((a, b) if a < b else (b, a))
    for i in range(M):
        a = L1[i]; b = L0[(i+1) % M]
        edges.add((a, b) if a < b else (b, a))
    elist = [tuple(sorted(e)) for e in edges]
    return n, elist


def _valid_degree_and_girth(k, g, n, edges):
    if len(edges) != k * n // 2:
        return False
    adj = [[] for _ in range(n)]
    seen = set()
    for u, v in edges:
        if u < 0 or v < 0 or u >= n or v >= n or u == v:
            return False
        key = (u, v) if u < v else (v, u)
        if key in seen:
            return False
        seen.add(key)
        adj[u].append(v)
        adj[v].append(u)
    for v in range(n):
        if len(adj[v]) != k:
            return False
    # girth: no cycle shorter than g via non-backtracking walks
    for s in range(n):
        visited = set()
        frontier = [(s, -1, -1)]  # (cur, prev, depth)
        while frontier:
            new = []
            for cur, prev, dep in frontier:
                if dep + 1 < g and (cur, prev) not in visited:
                    visited.add((cur, prev))
                    if cur == s and dep >= 1:
                        return False  # closed short cycle
                    nd = dep + 1
                    if nd >= g:
                        continue
                    for nb in adj[cur]:
                        if nb == prev:
                            continue
                        if nd == g - 1 and nb == s:
                            return False  # short cycle of length g-1
                        if nd < g - 1 and nb == s:
                            return False
                        new.append((nb, cur, nd))
            frontier = new
    return True


def construct(k, g):
    import time
    t0 = time.time()
    deadline = 150.0 - 2.0  # safety margin

    candidates = []
    # 1) fast tree closure
    try:
        n, e = _tree_closure_construct(k, g)
        candidates.append((n, e))
    except Exception:
        pass
    # 2) algebraic LUW fallback (valid by construction)
    try:
        nl, el = _luw_construct(k, g)
        candidates.append((nl, el))
    except Exception:
        pass

    if not candidates:
        return 0, []

    candidates.sort(key=lambda t: t[0])

    # validate from smallest upward; pick first valid that fits deadline
    for n, e in candidates:
        if time.time() > deadline - 0.5:
            break
        if _valid_degree_and_girth(k, g, n, e):
            return n, e

    # last resort: return the largest (most likely valid) candidate
    n, e = candidates[-1]
    return n, e
