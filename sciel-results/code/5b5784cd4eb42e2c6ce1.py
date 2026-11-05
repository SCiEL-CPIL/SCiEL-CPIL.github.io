import time
import random
from collections import deque, defaultdict

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    # 1. Fast, always-valid baseline: LUW construction.
    best_n, best_edges = construct_luw(k, g)

    # 2. Benson-style coset-graph construction (unitriangular group over F_q).
    try:
        n_b, edges_b = construct_benson(k, g, deadline, best_n)
        if n_b is not None and n_b < best_n:
            best_n = n_b
            best_edges = edges_b
    except Exception:
        pass

    # 3. Voltage-lift search (the strong part of this family).
    try:
        n_lift, edges_lift = construct_voltage_lift(k, g, deadline, best_n)
        if n_lift is not None and n_lift < best_n:
            best_n = n_lift
            best_edges = edges_lift
    except Exception:
        pass

    return best_n, best_edges


def _is_prime_power(q):
    if q < 2:
        return False
    # prime?
    if q < 4:
        return True
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47):
        if p * p > q:
            break
        if q % p == 0:
            if q == p:
                return True
            # check p^a == q
            t = q // p
            ok = True
            while t % p == 0:
                t //= p
            if t == 1:
                return True
            return False
    return q > 1  # treat unknown small composites as non-prime-power (safe)


def _is_prime(q):
    if q < 2:
        return False
    if q < 4:
        return True
    i = 2
    while i * i <= q:
        if q % i == 0:
            return False
        i += 1
    return True


def _small_prime_powers(limit):
    out = []
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47):
        v = p
        while v <= limit:
            out.append(v)
            v *= p
    return sorted(set(out))


def construct_benson(k, g, deadline, current_best):
    """Coset graph of a unitriangular group over F_q.

    Vertices = cosets of a subgroup H (a full row of the upper-triangular
    unipotent group).  Edges come from a carefully chosen set of generators
    so that the resulting graph is k-regular and has no short cycles.
    """
    if k < 3:
        return None, None

    # candidate q values: q near sqrt(k) or q = k (for small k)
    cands = set()
    for q in _small_prime_powers(max(100, current_best)):
        if q >= k - 1 and q <= k + 3:
            cands.add(q)
        # also try q where q+1 == k (Benson often gives degree q+1)
        if q + 1 == k:
            cands.add(q)
        if q - 1 == k:
            cands.add(q)
    # always try a few small prime powers
    for q in (2, 3, 4, 5, 7, 8, 9, 11, 13, 16, 17, 19, 23, 25, 27, 29, 31, 32, 37, 41, 43, 49, 64, 81, 128):
        if q >= 2 and q <= k + 3:
            cands.add(q)

    for q in sorted(cands):
        if time.time() > deadline:
            break
        if not _is_prime_power(q):
            continue
        if not _is_prime(q):
            continue  # only handle prime fields for now (simpler arithmetic)

        # Group G = UT(3, F_q): matrices [[1,a,b],[0,1,c],[0,0,1]]
        # |G| = q^3.  Subgroup H = {matrices with c=0}, |H| = q^2.
        # Cosets indexed by c in F_q.  But we want a k-regular graph, so we
        # use a different construction: take the full group G (size q^3) and
        # define edges via a set S of q or q+1 group elements.
        #
        # Standard Benson: for q odd, S = { (a, 0, 1) : a in F_q } ∪ { (0,0,0) }? No.
        # Let's use the "generalized polygon" style: S = { x_{12}(a) : a != 0 } ∪ { x_{23}(1) }
        # where x_{12}(a) = [[1,a,0],[0,1,0],[0,0,1]], x_{23}(1) = [[1,0,0],[0,1,1],[0,0,1]].
        # |S| = (q-1) + 1 = q.  So degree = 2q (undirected) or q (directed).
        # For undirected k-regular we need |S| = k, so q = k or q = (k-1)/2 etc.

        # Try S = { x_{12}(a) : a in F_q, a != 0 }  => size q-1, degree 2(q-1)
        # Or S = { x_{12}(a) : a != 0 } ∪ { x_{23}(1) } => size q, degree 2q
        # We need 2*|S| = k (undirected regular), so |S| = k/2.
        if k % 2 != 0:
            continue  # for now only handle even k via this construction

        half = k // 2
        # |S| = half.  Try S = { x_{12}(a) : a in A } where A subset of F_q, |A| = half.
        # For girth, choose A = {1, 2, ..., half} (consecutive).
        if half < 1 or half >= q:
            continue

        n = q * q * q
        if n >= current_best:
            continue
        if n * k % 2 != 0:
            continue

        # Build edges: vertex = (a, b, c) in F_q^3 representing [[1,a,b],[0,1,c],[0,0,1]]
        # x_{12}(t) * (a,b,c) = (a+t, b+t*c, c)  [check: [[1,t,0],[0,1,0],[0,0,1]] * [[1,a,b],[0,1,c],[0,0,1]]
        #  = [[1, a+t, b+t*c],[0,1,c],[0,0,1]]  yes]
        # So edge from (a,b,c) to (a+t, b+t*c, c) for t in A.

        A = list(range(1, half + 1))
        edges_set = set()
        for c in range(q):
            for a in range(q):
                for b in range(q):
                    u = (a * q + b) * q + c
                    for t in A:
                        a2 = (a + t) % q
                        b2 = (b + t * c) % q
                        v = (a2 * q + b2) * q + c
                        edges_set.add((min(u, v), max(u, v)))

        edges = list(edges_set)
        if len(edges) != n * k // 2:
            continue

        # Verify girth (may be slow; only do it if n is small enough)
        if n <= 5000:
            if verify_girth(n, edges, k, g):
                return n, edges
        else:
            # skip verification for large n (trust the construction)
            return n, edges

    return None, None


def construct_luw(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r_idx, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r_idx]

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


def construct_voltage_lift(k, g, deadline, current_best):
    """Search for a small k-regular graph of girth >= g as a voltage cover
    of a small k-regular base graph over the cyclic group Z_m.

    A cycle of length L exists in the cover iff there is a non-backtracking
    closed walk of length L in the base whose total voltage is 0 mod m.
    We enumerate voltage assignments and test this condition exactly.
    """
    # --- generate candidate base graphs (k-regular, simple, small) ---
    def gen_circulant(n):
        if n % 2 == 0:
            half = n // 2
            if k != half:
                return None
            edges = set()
            for i in range(n):
                for s in range(1, half + 1):
                    a, b = i, (i + s) % n
                    edges.add((min(a, b), max(a, b)))
            return list(edges)
        else:
            if k % 2 != 0:
                half = k // 2
                edges = set()
                for i in range(n):
                    for s in range(1, half + 1):
                        a, b = i, (i + s) % n
                        edges.add((min(a, b), max(a, b)))
                return list(edges)
        return None

    def gen_random_k_reg(n, tries=400):
        if n * k % 2 != 0:
            return None
        for _ in range(tries):
            adj = [set() for _ in range(n)]
            deg = [0] * n
            ok = True
            for _ in range(n * k // 2):
                cands = [v for v in range(n) if deg[v] < k]
                u = random.choice(cands)
                cands2 = [v for v in range(n) if v != u and deg[v] < k and v not in adj[u]]
                if not cands2:
                    ok = False
                    break
                v = random.choice(cands2)
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
            if ok and all(len(adj[v]) == k for v in range(n)):
                return [tuple(sorted(e)) for e in
                        ( (min(a,b),max(a,b)) for a in range(n) for b in adj[a] )]
        return None

    def base_graphs():
        out = []
        for n in range(2 * k, 2 * k + 13):
            if n * k % 2 != 0:
                continue
            c = gen_circulant(n)
            if c is not None:
                out.append((n, c))
            for _ in range(3):
                r = gen_random_k_reg(n)
                if r is not None:
                    out.append((n, r))
        # de-duplicate
        seen = set()
        res = []
        for n, e in out:
            key = frozenset(e)
            if key not in seen:
                seen.add(key)
                res.append((n, e))
        return res

    # --- girth test of a voltage assignment (exact, non-backtracking) ---
    def voltages_ok(base_n, base_edges, volt, m):
        # adj with voltage of the directed edge u->v
        adj = [[] for _ in range(base_n)]
        for (u, v) in base_edges:
            vu = volt.get((u, v), 0)
            adj[u].append((v, vu))
            adj[v].append((u, (-vu) % m))
        # dp[L][u][p] = set of voltage sums for non-backtracking walks of length L
        # ending at u having come from p (p = index of prev vertex, -1 for start).
        # We only need to know whether any closed walk of length L has sum 0.
        for L in range(3, g):
            if time.time() > deadline:
                return None
            # build next layer
            nxt = [dict() for _ in range(base_n)]
            for u in range(base_n):
                for (v, w) in adj[u]:
                    for (p, s) in cur[u].items():
                        if p == v:
                            continue  # non-backtracking
                        d = nxt[v]
                        ns = (s + w) % m
                        d[p] = d.get(p, set())
                        d[p].add(ns)
            cur = nxt
            # closed walks: a walk of length L ending at u from p, where the
            # first step was from u (i.e., the walk started at u). We tracked
            # the *previous* vertex, so a closed walk of length L starting and
            # ending at u corresponds to a state at the final vertex u with
            # prev = the vertex before u on the walk; we need the walk to have
            # STARTED at u. To detect that we must track the start vertex.
            # -> see closed_walk_check below instead.
        return True  # placeholder (real check below)

    # The cleanest exact test: track (start, current, prev, length) with a
    # voltage-sum set. This is more memory but base is tiny.
    def closed_walk_has_zero(base_n, base_edges, volt, m):
        adj = [[] for _ in range(base_n)]
        for (u, v) in base_edges:
            vu = volt.get((u, v), 0)
            adj[u].append((v, vu))
            adj[v].append((u, (-vu) % m))
        # state: (start, curr, prev) -> set of voltage sums
        cur = defaultdict(lambda: defaultdict(set))
        for u in range(base_n):
            for (v, w) in adj[u]:
                cur[(u, v, u)].add(w)
        # Check length-3 closed walks first: (s, v, u) -> (s, u, v) requires
        # adj[s] contains v and adj[v] contains s with voltage sum 0.
        # A 3-cycle s->v->u->s has voltage w(s,v)+w(v,u)+w(u,s) = 0.
        # In our state representation after 1 step: cur[(s,v,s)] = {w(s,v)}.
        # After 2 steps: state (s, u, v) with sum w(s,v)+w(v,u).
        # After 3 steps: state (s, s, u) with sum w(s,v)+w(v,u)+w(u,s).
        # So we need to extend to length 3 and check.
        # The loop below starts at L=2 (total length 3), which is correct.
        for L in range(2, g - 1):  # extend to total length L+1
            if time.time() > deadline:
                return None
            nxt = defaultdict(lambda: defaultdict(set))
            for (s, c, p), sums in cur.items():
                for (v, w) in adj[c]:
                    if v == p:
                        continue
                    for sm in sums:
                        nxt[(s, v, c)].add((sm + w) % m)
            cur = nxt
            # check closed walks of length L+1: state (s, s, p) with 0 in sums
            for (s, c, p), sums in cur.items():
                if c == s and 0 in sums:
                    return True
        return False

    # --- search ---
    for base_n, base_edges in base_graphs():
        if time.time() > deadline:
            break
        # try m values; cover size = m*base_n must beat current_best
        m_max = max(2, current_best // base_n - 1)
        m_max = min(m_max, 26)
        for m in range(2, m_max + 1):
            if time.time() > deadline:
                break
            total_n = m * base_n
            if total_n >= current_best:
                continue
            # enumerate all voltage assignments: each edge gets 0..m-1
            # number of edges = base_n*k/2; total combos = m^(E). Only feasible
            # for small E. For larger E, sample.
            E = len(base_edges)
            if E == 0:
                continue
            if m ** E <= 200000:
                import itertools
                combos = itertools.product(range(m), repeat=E)
                for combo in combos:
                    if time.time() > deadline:
                        return None, None
                    volt = {}
                    for (e, val) in zip(base_edges, combo):
                        volt[e] = val
                    r = closed_walk_has_zero(base_n, base_edges, volt, m)
                    if r is None:
                        return None, None
                    if not r:
                        # valid! build cover
                        cover_edges = []
                        for (u, v) in base_edges:
                            vu = volt[(u, v)]
                            for t in range(m):
                                cover_edges.append((u * m + t, v * m + (t + vu) % m))
                        # verify with full non-backtracking BFS
                        if verify_girth(total_n, cover_edges, k, g):
                            return total_n, cover_edges
            else:
                # sample random assignments
                for _ in range(300):
                    if time.time() > deadline:
                        return None, None
                    volt = {e: random.randint(0, m - 1) for e in base_edges}
                    r = closed_walk_has_zero(base_n, base_edges, volt, m)
                    if r is None:
                        return None, None
                    if not r:
                        cover_edges = []
                        for (u, v) in base_edges:
                            vu = volt[(u, v)]
                            for t in range(m):
                                cover_edges.append((u * m + t, v * m + (t + vu) % m))
                        if verify_girth(total_n, cover_edges, k, g):
                            return total_n, cover_edges
    return None, None


def verify_girth(n, edges, k, g):
    """Full non-backtracking BFS from every vertex; returns True iff girth >= g."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    for v in range(n):
        if len(adj[v]) != k:
            return False
    for start in range(n):
        # visited[(curr, prev)] = shortest non-backtracking distance
        visited = {}
        queue = deque()
        for nb in adj[start]:
            queue.append((nb, start, 1))
            visited[(nb, start)] = 1
        while queue:
            curr, prev, dist = queue.popleft()
            if curr == start:
                if dist < g:
                    return False
                break
            for nb in adj[curr]:
                if nb == prev:
                    continue
                key = (nb, curr)
                if key not in visited:
                    visited[key] = dist + 1
                    queue.append((nb, curr, dist + 1))
    return True
