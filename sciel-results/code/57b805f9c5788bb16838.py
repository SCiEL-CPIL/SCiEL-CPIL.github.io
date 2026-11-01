import time
import numpy as np

_T0 = time.time()
_LIMIT = 150.0
_MARGIN = 18.0
_DEADLINE = _T0 + (_LIMIT - _MARGIN)


def _now():
    return time.time()


def _time_left():
    return _DEADLINE - _now()


def _valid(n, edges, k, g):
    """Cheap structural checks (degree, simple, connected)."""
    if n < 1:
        return False
    deg = [0] * n
    seen = set()
    for a, b in edges:
        if a < 0 or b < 0 or a >= n or b >= n or a == b:
            return False
        if a > b:
            a, b = b, a
        if (a, b) in seen:
            return False
        seen.add((a, b))
        deg[a] += 1
        deg[b] += 1
    for d in deg:
        if d != k:
            return False
    # connectivity (BFS)
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    stack = [0]
    vis = [False] * n
    vis[0] = True
    cnt = 1
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if not vis[v]:
                vis[v] = True
                cnt += 1
                stack.append(v)
    return cnt == n


def _girth_bfs(n, edges, g):
    """Return the true girth via BFS from every vertex (certifying >= g)."""
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    best = None
    for s in range(n):
        dist = [-1] * n
        par = [-1] * n
        dist[s] = 0
        q = [s]
        head = 0
        while head < len(q):
            u = q[head]
            head += 1
            du = dist[u]
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    par[v] = u
                    q.append(v)
                elif v != par[u]:
                    c = du + dist[v] + 1
                    if best is None or c < best:
                        best = c
                        if best < g:
                            return best
        if best is not None and best < g:
            return best
    return best if best is not None else 10 ** 9


def _girth_circ(n, S, g):
    """Girth of circulant Circ(n; S) by signed-sum BFS (state = residue)."""
    S = list(S)
    if not S:
        return 10 ** 9
    m = len(S)
    dist = [-1] * n
    dist[0] = 0
    q = [0]
    head = 0
    while head < len(q):
        r = q[head]
        head += 1
        d = dist[r]
        if d >= g:
            break
        nd = d + 1
        for s in S:
            a = (r + s) % n
            b = (r - s) % n
            for t in (a, b):
                if dist[t] == -1:
                    dist[t] = nd
                    q.append(t)
    # cycle exists iff some nontrivial walk returns to 0 with length >= 2
    # find min length L>=2 with a residue 0 reachable at exactly L
    # (dist[0]==0 is trivial; we need the next return)
    # recompute with parent to detect return
    # simpler: BFS but allow revisiting 0 to measure first return
    dist2 = [-1] * n
    dist2[0] = 0
    q2 = [0]
    head2 = 0
    found = None
    while head2 < len(q2):
        r = q2[head2]
        head2 += 1
        d = dist2[r]
        if d >= g:
            break
        nd = d + 1
        for s in S:
            for t in ((r + s) % n, (r - s) % n):
                if t == 0:
                    if nd >= 2:
                        if found is None or nd < found:
                            found = nd
                elif dist2[t] == -1:
                    dist2[t] = nd
                    q2.append(t)
    return found if found is not None else 10 ** 9


def _circulant_edges(n, S):
    edges = []
    S = list(S)
    for i in range(n):
        for s in S:
            j = (i + s) % n
            if i < j:
                edges.append((i, j))
    return edges


def _circulant_search(k, g, budget):
    k_half = k // 2
    best = None
    # candidate moduli
    moduli = []
    for n in range(k * (g - 1), 4 * k * (g - 1) + 1):
        moduli.append(n)
    for n in range(2 * k * (g - 1), 6 * k * (g - 1) + 1):
        if n not in moduli:
            moduli.append(n)
    moduli = sorted(set(moduli))
    for n in moduli:
        if _time_left() < 5:
            break
        # try random sets of size k_half
        for trial in range(200):
            if _time_left() < 5:
                break
            S = list(np.random.choice(n, size=k_half, replace=False))
            gt = _girth_circ(n, S, g)
            if gt >= g:
                if best is None or n < best[0]:
                    best = (n, S)
                break
    if best is not None:
        n, S = best
        return n, _circulant_edges(n, S)
    return None


def _lift_search(k, g, budget):
    """Build a k-regular base (circulant) then 2-lift it to raise girth."""
    k_half = k // 2
    # base girth target: lift doubles, so base girth ~ g is enough; require >= g-1
    base = _circulant_search(k, max(4, g - 1), budget)
    if base is None:
        return None
    bn, bedges = base
    badj = [[] for _ in range(bn)]
    for a, b in bedges:
        badj[a].append(b)
        badj[b].append(a)
    # random 2-lift via signs on edges
    for trial in range(3000):
        if _time_left() < 6:
            break
        sign = np.random.randint(0, 2, size=len(bedges))
        # build lifted adjacency
        ln = 2 * bn
        ladj = [[] for _ in range(ln)]
        for idx, (a, b) in enumerate(bedges):
            s = sign[idx]
            # a0 - b(s), a1 - b(1-s)
            ladj[a * 2 + 0].append(b * 2 + s)
            ladj[b * 2 + s].append(a * 2 + 0)
            ladj[a * 2 + 1].append(b * 2 + (1 - s))
            ladj[b * 2 + (1 - s)].append(a * 2 + 1)
        # girth check via BFS on lifted graph (stop early at g)
        best = None
        ok = True
        for s0 in range(ln):
            dist = [-1] * ln
            par = [-1] * ln
            dist[s0] = 0
            q = [s0]
            head = 0
            while head < len(q):
                u = q[head]
                head += 1
                du = dist[u]
                if du >= g:
                    break
                for v in ladj[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        par[v] = u
                        q.append(v)
                    elif v != par[u]:
                        c = du + dist[v] + 1
                        if c < g:
                            ok = False
                            break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            # success: build edge list
            edges = []
            for a in range(ln):
                for b in ladj[a]:
                    if a < b:
                        edges.append((a, b))
            return ln, edges
    return None


def _greedy_girth(n, edges, g):
    """Try to add edges (keeping k-regularity via swaps) to raise girth.
    Simple version: if current girth already >= g, optionally try to *reduce*
    n by edge swaps is hard; instead we just verify. Returns edges unchanged
    but we use it as a polish: attempt 2-opt edge swaps that keep degree and
    never lower girth, to enable a smaller construction later. Here we keep it
    minimal: return edges as-is (girth already certified)."""
    return edges


def construct(k, g):
    global _DEADLINE
    _DEADLINE = _now() + max(1.0, _LIMIT - _MARGIN)
    rng_seed = 12345
    np.random.seed(rng_seed)

    best = None

    # Strategy A: circulant (fast, often good for moderate girth)
    res = _circulant_search(k, g, 20)
    if res is not None and _valid(res[0], res[1], k, g):
        gt = _girth_bfs(res[0], res[1], g)
        if gt >= g:
            best = res

    # Strategy B: 2-lift (can beat plain circulant)
    if _time_left() > 20:
        res2 = _lift_search(k, g, 30)
        if res2 is not None and _valid(res2[0], res2[1], k, g):
            gt = _girth_bfs(res2[0], res2[1], g)
            if gt >= g:
                if best is None or res2[0] < best[0]:
                    best = res2

    # Fallback: seed's LUW construction (guaranteed valid, large)
    if best is None:
        best = _luw(k, g)

    # Final certification
    n, edges = best
    gt = _girth_bfs(n, edges, g)
    if gt < g or not _valid(n, edges, k, g):
        # safety: regenerate with LUW
        n, edges = _luw(k, g)
    return n, edges


def _luw(k, g):
    """The original Lazebnik-Ustimenko-Woldar construction (fallback)."""
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
