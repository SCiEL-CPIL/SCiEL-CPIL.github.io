import time
import random
from collections import deque, defaultdict

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    # 1. Fast, always-valid baseline: LUW construction.
    best_n, best_edges = construct_luw(k, g)

    # 2. Voltage-lift search (the strong part of this family).
    try:
        n_lift, edges_lift = construct_voltage_lift(k, g, deadline, best_n)
        if n_lift is not None and n_lift < best_n:
            best_n = n_lift
            best_edges = edges_lift
    except Exception:
        pass

    return best_n, best_edges


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
