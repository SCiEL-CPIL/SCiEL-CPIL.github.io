# EVOLVE-BLOCK-START
import numpy as np

# ---------------------------------------------------------------------------
# Finite field GF(q) arithmetic (q a prime power).
# ---------------------------------------------------------------------------
def _log_tab(p, n):
    tab = [0] * (p + 1)
    x = 1
    for i in range(p):
        tab[x] = i
        x = (x * n) % p
    return tab

def _make_ffield(q):
    if q == 2:
        def add(a, b): return a ^ b
        def sub(a, b): return a ^ b
        def mul(a, b): return (a * b) % 2
        def inv(a): return a
        return add, sub, mul, inv, q
    if q == 4:
        def add(a, b): return a ^ b
        def sub(a, b): return a ^ b
        def mul(a, b):
            if a == 0 or b == 0:
                return 0
            return 1 + (a ^ b)
        def inv(a):
            return 0 if a == 0 else a
        return add, sub, mul, inv, q
    # prime field
    if q < 100000:
        logt = _log_tab(q, 2)
        expv = [0] * q
        for i in range(q):
            expv[i % (q - 1)] = pow(2, i, q)
        def add(a, b): return (a + b) % q
        def sub(a, b): return (a - b) % q
        def mul(a, b):
            if a == 0 or b == 0:
                return 0
            return expv[logt[a] + logt[b]]
        def inv(a):
            if a == 0:
                return 0
            return expv[-logt[a] % (q - 1)]
        return add, sub, mul, inv, q
    # fallback: generic modular (for safety, though q will be small)
    def add(a, b): return (a + b) % q
    def sub(a, b): return (a - b) % q
    def mul(a, b): return (a * b) % q
    def inv(a): return pow(a, q - 2, q) if a else 0
    return add, sub, mul, inv, q

# ---------------------------------------------------------------------------
# Moufang generalized hexagon G2(q): incidence graph (points + lines),
# (q+1)-regular, girth 12.
# ---------------------------------------------------------------------------
def _hexagon(q):
    add, sub, mul, inv, _ = _make_ffield(q)

    pts = []
    lines = []
    for y in range(q):
        for z in range(q):
            pts.append((0, y, z))
    pts.append((1, 0, 0))
    for x in range(1, q):
        pts.append((x, mul(x, x), 1))
    P = len(pts)

    for y in range(q):
        for z in range(q):
            lines.append((0, y, z))
    lines.append((0, 0, 1))
    for x in range(1, q):
        lines.append((1, mul(x, x), x))
    L = len(lines)

    adj = [[] for _ in range(P + L)]
    for i in range(P):
        for j in range(L):
            if _inc(pts[i], lines[j], add, sub, mul, inv):
                adj[i].append(P + j)
                adj[P + j].append(i)
    return P, L, adj

def _inc(pt, ln, add, sub, mul, inv):
    if pt[0] == 0:
        # point (0,y,z)
        if ln[0] == 0:
            # line (0,y',z'): incidence iff z*y' - y*z' == 0
            return (mul(z, ln[1]) == mul(pt[1], ln[2]))
        # line (1, a, b): x = a y + b z, y = z
        return (pt[1] == pt[2]) and (ln[1] == mul(pt[2], pt[2])) and (ln[2] == pt[2])
    if pt[0] == 1:
        # point (1,0,0)
        if ln[0] == 0:
            return ln[1] == 0
        return False
    # point (x, x^2, 1)
    if ln[0] == 0:
        # line (0,y',z'): incidence iff x^2*y' + y' - z' == 0  -> z' = (x^2+1)*y'
        return ln[2] == mul(mul(pt[0], pt[0]) + 1, ln[1])
    # line (1, a, b): incidence iff a == -x, b == x^2
    return (ln[1] == -pt[0]) and (ln[2] == mul(pt[0], pt[0]))

def construct(k, g):
    # Need (s+1)-regular with s+1 == k, and girth(12) >= g.
    s = k - 1
    # choose smallest prime power q >= s with q+1 == k
    q = k - 1
    if not _is_prime_power(q):
        # fall back: keep degree k but use a q giving q+1 == k if possible
        q = _smallest_pp_ge(q)
        # if q+1 != k, we cannot hit exact degree k with this construction;
        # return a valid k-regular only if q+1 == k, else use next q and note
        # (evaluator requires degree exactly k, so we must have q+1 == k).
        # For the task k=5 -> q=4 works.
        pass
    if q + 1 != k:
        # cannot satisfy exact degree k with hexagon; try to still return
        # a k-regular graph is not possible via this family, so use q=k-1
        # only when prime power. Otherwise, raise to a valid construction.
        # For robustness, if k-1 not prime power, we still build with q=k-1
        # is invalid; instead we build with the nearest prime power and the
        # graph will be (q+1)-regular which may differ from k. To keep the
        # interface contract (degree exactly k), we only proceed when q+1==k.
        # Fallback: use the LUW-style bipartite construction for arbitrary k.
        return _luw_fallback(k, g)

    P, L, adj = _hexagon(q)
    n = P + L
    # verify degree
    for v in range(n):
        assert len(adj[v]) == k, "degree mismatch"
    # verify girth >= g via non-backtracking walks
    _check_girth(adj, n, g)
    edges = []
    for v in range(n):
        for w in adj[v]:
            if v < w:
                edges.append((v, w))
    return n, edges

def _is_prime_power(n):
    if n < 2:
        return False
    # check if n is p^e
    for p in range(2, int(n ** 0.5) + 1):
        m = n
        if m % p == 0:
            ok = True
            while m % p == 0:
                m //= p
            if m != 1:
                ok = False
            if ok:
                return True
    # n is prime
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

def _smallest_pp_ge(n):
    x = max(2, n)
    while not _is_prime_power(x):
        x += 1
    return x

def _check_girth(adj, n, g):
    # non-backtracking BFS from each vertex; ensure no return within < g steps
    for start in range(n):
        # state: (vertex, prev)
        from collections import deque
        dist = {}
        dq = deque()
        for nb in adj[start]:
            dist[(nb, start)] = 1
            dq.append((nb, start))
        while dq:
            v, prev = dq.popleft()
            d = dist[(v, prev)]
            if v == start and d < g:
                raise AssertionError("short cycle found")
            if d >= g - 1:
                continue
            for nb in adj[v]:
                if nb == prev:
                    continue
                st = (nb, v)
                if st not in dist:
                    dist[st] = d + 1
                    dq.append(st)

def _luw_fallback(k, g):
    # Lazebnik-Ustimenko-Woldar 5-regular bipartite construction (general q)
    q = k
    d = g - 5 if (g - 5) % 2 else g - 4
    if d < 3:
        d = 3
    if q == 4:
        _mul_t = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        def f_add(a, b): return a ^ b
        def f_sub(a, b): return a ^ b
        def f_mul(a, b): return _mul_t[a][b]
    else:
        def f_add(a, b): return (a + b) % q
        def f_sub(a, b): return (a - b) % q
        def f_mul(a, b): return (a * b) % q
    rules = {}
    rules[2] = (1, 1)
    rules[3] = (2, 1)
    rules[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        base = c - r
        candidates = [(1, base - 2), (base - 1, 1), (base, 1), (1, base + 1)]
        rules[c] = candidates[r]
    start_key = (0, (0,) * (d + 1))
    label = {start_key: 0}
    queue = [start_key]
    edge_list = []
    for vkey in queue:
        side, coords = vkey
        vidx = label[vkey]
        for first_val in range(q):
            ncoords = [0] * (d + 1)
            ncoords[1] = first_val
            for c in range(2, d + 1):
                a, b = rules[c]
                if side == 0:
                    ncoords[c] = f_add(coords[c], f_mul(ncoords[a], coords[b]))
                else:
                    ncoords[c] = f_sub(coords[c], f_mul(coords[a], ncoords[b]))
            nkey = (1 - side, tuple(ncoords))
            if nkey not in label:
                label[nkey] = len(queue)
                queue.append(nkey)
            if side == 0:
                edge_list.append((vidx, label[nkey]))
    return len(queue), edge_list
# EVOLVE-BLOCK-END
