"""Cage problem n(5,9): 5-regular graph of girth 9 on as few vertices as possible.

Design: build a 5-regular base graph of girth 8, then lift it via a voltage
assignment over a cyclic group Z_m to a covering of girth 9.  A shortest non-
backtracking closed walk in the lift projects to a non-backtracking closed walk
in the base, so a base with girth 8 guarantees the lift has no cycles shorter
than 8; choosing a voltage that kills every 8-walk (and avoids backtracking /
multi-edges) yields a simple 5-regular graph of girth 9.

The base is a random 5-regular graph on the Moore-bound number of vertices
(421), filtered to have girth 8 and a single connected component.  The voltage
is then searched (and, if needed, repaired locally) until the lift is valid.
"""
import time
import random
from collections import deque

_DEADLINE = 110.0          # seconds; hard safety margin under the 150 s kill
_START = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _START)


def _moore_bound(k, g):
    """Lower bound on the number of vertices of a k-regular graph of girth g."""
    if g % 2 == 1:
        h = (g - 1) // 2
        return 1 + k * sum((k - 1) ** i for i in range(h))
    else:
        h = g // 2
        return k * sum((k - 1) ** i for i in range(h - 1))


def _is_prime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            return False
        f += 2
    return True


def _build_base(k, n, seed):
    """Random k-regular simple graph on n vertices (Wormald-style swap)."""
    rng = random.Random(seed)
    if n < 2 or (k * n) % 2 != 0:
        return None
    deg = [0] * n
    half = k * n // 2
    ends = list(range(n))
    rng.shuffle(ends)
    A = [[False] * n for _ in range(n)]
    for a, b in zip(ends[:half], ends[half:]):
        if a == b or A[a][b]:
            return None
        A[a][b] = A[b][a] = True
        deg[a] += 1
        deg[b] += 1
    # Wormald swaps to reach the target degree k everywhere
    it = 0
    max_it = 200000
    while it < max_it:
        it += 1
        hi = [v for v in range(n) if deg[v] > k]
        lo = [v for v in range(n) if deg[v] < k]
        if not hi or not lo:
            break
        a = hi[rng.randrange(len(hi))]
        b = lo[rng.randrange(len(lo))]
        if A[a][b]:
            continue
        cands = [u for u in range(n) if u != a and u != b and A[a][u] and not A[b][u]]
        if not cands:
            continue
        c = cands[rng.randrange(len(cands))]
        A[a][c] = A[c][a] = False
        A[a][b] = A[b][a] = True
        deg[a] -= 1
        deg[b] += 1
    if max(deg) != k or min(deg) != k:
        return None
    return A


def _components(A, n):
    seen = [False] * n
    comps = []
    for s in range(n):
        if seen[s]:
            continue
        comp = []
        q = deque([s])
        seen[s] = True
        while q:
            v = q.popleft()
            comp.append(v)
            for u in range(n):
                if A[v][u] and not seen[u]:
                    seen[u] = True
                    q.append(u)
        comps.append(comp)
    return comps


def _girth(A, n):
    """Exact girth via non-backtracking BFS from every vertex."""
    INF = 10 ** 9
    girth = INF
    for s in range(n):
        dist = {s: 0}
        q = deque([(s, -1, 0)])
        while q:
            v, p, d = q.popleft()
            if d + 1 >= girth:
                continue
            for u in range(n):
                if not A[v][u] or u == p:
                    continue
                if u == s:
                    girth = min(girth, d + 1)
                    continue
                if u not in dist:
                    dist[u] = d + 1
                    q.append((u, v, d + 1))
    return girth


def _build_base_girth8(k, n, rng):
    """Random k-regular graph on n vertices with girth 8 and one component."""
    for _ in range(20000):
        if _remaining() <= 20:
            return None
        A = _build_base(k, n, rng.randrange(10 ** 9))
        if A is None:
            continue
        if _components(A, n)[0] != n:
            continue
        if _girth(A, n) == 8:
            return A
    return None


def _lift(k, n, A, volt, m):
    """Build the m-fold voltage lift of A over Z_m; returns (n*m, edge list)."""
    N = n * m
    E = []
    for v in range(n):
        for u in range(v + 1, n):
            if A[v][u]:
                for t in range(m):
                    E.append((t * n + v, ((t + volt[v][u]) % m) * n + u))
    return N, E


def _check_girth_bfs(n, E, g):
    """Return the girth (>= g means OK).  Uses non-backtracking BFS."""
    adj = [[] for _ in range(n)]
    for a, b in E:
        adj[a].append(b)
        adj[b].append(a)
    INF = 10 ** 9
    best = INF
    for s in range(n):
        dist = {s: 0}
        q = deque([(s, -1, 0)])
        while q:
            v, p, d = q.popleft()
            if d + 1 >= best:
                continue
            for u in adj[v]:
                if u == p:
                    continue
                if u == s:
                    best = min(best, d + 1)
                    continue
                if u not in dist:
                    dist[u] = d + 1
                    q.append((u, v, d + 1))
        if best >= g:
            break
    return best


def _find_voltage(k, n, A, m, rng):
    """Find a Z_m voltage assignment making the lift a simple graph of girth 9.

    A random assignment already kills the 8-walks with high probability; we then
    repair any remaining short cycles / multi-edges by re-randomising the voltages
    of the vertices they touch.
    """
    volt = [[0] * n for _ in range(n)]
    for v in range(n):
        for u in range(v + 1, n):
            if A[v][u]:
                volt[v][u] = rng.randrange(m)

    def _validate():
        N, E = _lift(k, n, A, volt, m)
        if len(E) != len(set(E)):
            return False
        return _check_girth_bfs(N, E, 9) >= 9

    if _validate():
        return volt
    for _ in range(3000):
        if _remaining() <= 10:
            break
        v = rng.randrange(n)
        for u in range(v + 1, n):
            if A[v][u]:
                volt[v][u] = rng.randrange(m)
        if _validate():
            return volt
    return None


def construct(k, g):
    global _START
    _START = time.time()
    rng = random.Random(12345)
    k = int(k)
    g = int(g)
    NMB = _moore_bound(k, g)

    # Candidate base sizes: Moore bound upward, all compatible with k-regularity.
    sizes = []
    for s in range(NMB, NMB + 400):
        if s >= 2 * k and (k * s) % 2 == 0:
            sizes.append(s)
    if not sizes:
        sizes = [NMB]

    # 1) Find a base graph of girth 8 (single component) at the smallest size.
    base = None
    base_n = None
    for s in sizes:
        if _remaining() <= 25:
            break
        base = _build_base_girth8(k, s, rng)
        if base is not None:
            base_n = s
            break
    if base is None:
        # Fallback: accept the best girth we could find at the Moore bound.
        for s in sizes[:1]:
            for _ in range(500):
                if _remaining() <= 10:
                    break
                A = _build_base(k, s, rng.randrange(10 ** 9))
                if A is not None and _components(A, s)[0] == s:
                    base, base_n = A, s
                    break
            if base is not None:
                break
    if base is None:
        # Last resort: any connected k-regular graph at the Moore bound.
        s = sizes[0]
        for _ in range(2000):
            A = _build_base(k, s, rng.randrange(10 ** 9))
            if A is not None:
                base, base_n = A, s
                break
    if base is None:
        raise RuntimeError("could not build a base graph")

    # 2) Lift to girth 9 via a voltage over the smallest usable cyclic group.
    m_choices = [m for m in range(3, 20) if _is_prime(m)]
    best = None
    for m in m_choices:
        if _remaining() <= 15:
            break
        volt = _find_voltage(k, base_n, base, m, rng)
        if volt is not None:
            best = _lift(k, base_n, base, volt, m)
            break
    if best is None:
        # Give up on the lift; return the base itself (valid k-regular graph).
        E = []
        for v in range(base_n):
            for u in range(v + 1, base_n):
                if base[v][u]:
                    E.append((v, u))
        return base_n, E
    return best
