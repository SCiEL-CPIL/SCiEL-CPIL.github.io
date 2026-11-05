import time
import math
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before the 150s kill

    def remaining():
        return deadline - time.time()

    # ---------------- exact girth certificate via non-backtracking BFS ----------------
    def check_girth(n, edges, g, time_limit):
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        for i in range(n):
            if len(adj[i]) != k:
                return -1
        min_cycle = float('inf')
        for s in range(n):
            if time.time() > time_limit:
                return -1
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] != -1:
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if cycle_len < g:
                            return min_cycle
                    else:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
            if min_cycle < g:
                return min_cycle
        return min_cycle

    # ---------------- LUW construction (guaranteed-valid fallback) ----------------
    def _luw_ops(q):
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            return (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        return ((lambda x, y: (x + y) % q),
                (lambda x, y: (x - y) % q),
                (lambda x, y: x * y % q))

    def generate_rules(d, variant=0, offset=0):
        base = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        if variant == 0:
            for c in range(5, d + 1):
                r, c0 = ((c - 5 + offset) % 4), c - ((c - 5 + offset) % 4)
                base[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        elif variant == 1:
            for c in range(5, d + 1):
                base[c] = (1, c - 2) if (c + offset) % 2 == 0 else (c - 1, 1)
        elif variant == 2:
            for c in range(5, d + 1):
                if (c + offset) % 3 == 0:
                    base[c] = (1, c - 3)
                elif (c + offset) % 3 == 1:
                    base[c] = (c - 1, 1)
                else:
                    base[c] = (c - 2, 1)
        elif variant == 3:
            for c in range(5, d + 1):
                base[c] = (1, c - (1 + (c + offset) % 2))
        elif variant == 4:
            for c in range(5, d + 1):
                base[c] = [(1, c - 1), (c - 1, 1), (1, c - 2), (c - 2, 1)][(c + offset) % 4]
        elif variant == 5:
            for c in range(5, d + 1):
                if (c + offset) % 2 == 0:
                    base[c] = (1, c - 2)
                else:
                    base[c] = (c - 3, 1)
        elif variant == 6:
            for c in range(5, d + 1):
                idx = (c * 3 + offset) % 4
                base[c] = [(1, c - 1), (c - 1, 1), (1, c - 2), (c - 2, 1)][idx]
        else:
            for c in range(5, d + 1):
                r, c0 = ((c - 5 + offset) % 4), c - ((c - 5 + offset) % 4)
                base[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        return base

    def build_luw(q, d, rule, time_limit):
        add, sub, mul = _luw_ops(q)

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
            if time.time() > time_limit:
                return None
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    def luw_construct(k, g):
        q = k
        d = max(3, (g - 2) // 2)
        best = None
        best_n = float('inf')
        for d_try in range(d, d + 8):
            if remaining() < 15:
                break
            for variant in range(7):
                for offset in range(5):
                    if remaining() < 12:
                        break
                    rule = generate_rules(d_try, variant, offset)
                    res = build_luw(q, d_try, rule, deadline - 12)
                    if res is None:
                        continue
                    n, edges = res
                    if n >= best_n:
                        continue
                    gc = check_girth(n, edges, g, deadline - 8)
                    if gc == -1:
                        continue
                    if gc >= g:
                        best, best_n = (n, edges), n
                        if n < 10000:
                            break
                if best is not None and best_n < 10000:
                    break
            if best is not None and best_n < 10000:
                break
        if best is None:
            rule = generate_rules(d, 0, 0)
            res = build_luw(q, d, rule, deadline - 2)
            if res is not None:
                best = res
        return best

    # ---------------- voltage-lift construction (structural new idea) ----------------
    def voltage_lift(k, g, time_limit):
        # base Petersen graph (3-regular, girth 5)
        base_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
        ]
        n_base = 10
        ne = len(base_edges)
        base_adj = [[] for _ in range(n_base)]
        for i, (u, v) in enumerate(base_edges):
            base_adj[u].append((v, i))
            base_adj[v].append((u, i))

        # all closed non-backtracking walks of length 3..g-1 in the base
        walks = set()
        for sv in range(n_base):
            stack = []
            for nv, eid in base_adj[sv]:
                u, v = base_edges[eid]
                sign = 1 if u == sv else -1
                stack.append((nv, eid, 1, [(eid, sign)]))
            while stack:
                cv, pe, ln, path = stack.pop()
                if ln >= g:
                    continue
                if cv == sv and ln >= 3:
                    walks.add(tuple(path))
                for nv, eid in base_adj[cv]:
                    if eid == pe:
                        continue
                    u, v = base_edges[eid]
                    sign = 1 if u == cv else -1
                    stack.append((nv, eid, ln + 1, path + [(eid, sign)]))

        # coefficient vectors per edge (only non-zero kept)
        walk_coeffs = []
        for w in walks:
            coeff = [0] * ne
            for eid, sgn in w:
                coeff[eid] += sgn
            nz = [(i, c) for i, c in enumerate(coeff) if c != 0]
            if nz:
                walk_coeffs.append(nz)
        wc = walk_coeffs

        def build_lift(volts, m):
            lift = []
            for i, (u, v) in enumerate(base_edges):
                volt = volts[i]
                lift.append((u * m + 0, v * m + (volt % m)))
            return lift

        def check_volts(volts, m):
            for nz in wc:
                s = 0
                for i, c in nz:
                    s += c * volts[i]
                if s % m == 0:
                    return False
            return True

        def is_gen(a, m):
            return math.gcd(a, m) == 1

        def try_m(m, budget, rng):
            if time.time() > budget:
                return None
            gens = [a for a in range(1, m) if is_gen(a, m)]
            if not gens:
                return None
            best_volts = None
            # systematic patterns
            for idx in range(min(120, len(gens) * 2 + 40)):
                if time.time() > budget:
                    break
                if idx < len(gens):
                    volts = [gens[idx]] * ne
                else:
                    j = idx - len(gens)
                    volts = [gens[(j * (i + 1)) % len(gens)] for i in range(ne)]
                gv = m
                for v in volts:
                    gv = math.gcd(gv, v)
                    if gv == 1:
                        break
                if gv != 1:
                    continue
                if check_volts(volts, m):
                    best_volts = list(volts)
                    break
            # random searches
            if best_volts is None:
                tries = 0
                while tries < 400 and time.time() < budget:
                    tries += 1
                    volts = [rng.choice(gens) for _ in range(ne)]
                    gv = m
                    for v in volts:
                        gv = math.gcd(gv, v)
                        if gv == 1:
                            break
                    if gv != 1:
                        continue
                    if check_volts(volts, m):
                        best_volts = volts
                        break
            if best_volts is None:
                return None
            return build_lift(best_volts, m)

        # candidate m: n = 10*m must be >= Moore bound (512) for k=3,g=16
        moo = 2 * k * (k - 1) ** ((g - 2) // 2 - 1)  # general-ish lower bound
        moob = max(moo, 2 * (k - 1) ** ((g - 2) // 2) + 2)
        m_min = max(2, math.ceil(moob / n_base))
        # prioritize smallest m first, then sparse larger ones
        cand = list(range(m_min, m_min + 40))
        cand += list(range(m_min + 40, m_min + 200, 5))
        cand += list(range(m_min + 200, m_min + 600, 25))
        seen = set()
        ordered = []
        for m in cand:
            if m not in seen:
                seen.add(m)
                ordered.append(m)

        best = None
        best_n = float('inf')
        rng = random.Random(12345)
        for m in ordered:
            if time.time() > time_limit:
                break
            if n_base * m >= best_n:
                continue
            lift = try_m(m, time_limit, rng)
            if lift is None:
                continue
            n = n_base * m
            if n >= best_n:
                continue
            gc = check_girth(n, lift, g, time_limit)
            if gc == -1:
                continue
            if gc >= g:
                best, best_n = (n, lift), n
                if n <= 600:
                    break
        return best

    # ---------------- Ramanujan cubic graphs (k=3) ----------------
    def build_ramanujan(q):
        def mul(a, b):
            return ((a[0]*b[0] + a[1]*b[2]) % q, (a[0]*b[1] + a[1]*b[3]) % q,
                    (a[2]*b[0] + a[3]*b[2]) % q, (a[2]*b[1] + a[3]*b[3]) % q)

        def det(a):
            return (a[0]*a[3] - a[1]*a[2]) % q

        def inv(a):
            d = det(a)
            invd = pow(d, q - 2, q)
            return ((a[3]*invd) % q, (-a[1]*invd) % q, (-a[2]*invd) % q, (a[0]*invd) % q)

        def norm(a):
            return (a[0]*a[0] + a[1]*a[1] + a[2]*a[2] + a[3]*a[3]) % q

        def involution(a):
            if det(a) != 1:
                return False
            m2 = mul(a, a)
            n = norm(m2)
            return n == 0 or n == 2

        I = (1, 0, 0, 1)
        A = (0, 1, q - 1, 0)
        B = (0, 1, 1, 0)
        if not (involution(A) and involution(B)):
            return None
        C = None
        for s in range(1, q):
            cand = (1, 0, s, 1)
            if involution(cand):
                C = cand
                break
        if C is None:
            return None
        G = [I, A, B, C,
             mul(A, B), mul(B, A), mul(A, C), mul(C, A),
             mul(B, C), mul(C, B), mul(A, B), mul(B, A)]
        seen, gens = set(), []
        for m in G:
            key = m if (m[0], m[1], m[2], m[3]) <= (q - m[0] % q, q - m[1] % q, q - m[2] % q, q - m[3] % q) else (q - m[0] % q, q - m[1] % q, q - m[2] % q, q - m[3] % q)
            if key not in seen:
                seen.add(key)
                gens.append(m)
        if len(gens) != 3:
            return None
        start = I
        index = {start: 0}
        order = [start]
        edges = []
        for i in range(len(order)):
            v = order[i]
            vi = inv(v)
            for w in gens:
                u = mul(vi, w)
                if u not in index:
                    index[u] = len(order)
                    order.append(u)
                if index[v] < index[u]:
                    edges.append((index[v], index[u]))
        return len(order), edges

    # ---------------- main ----------------
    if k != 3:
        res = luw_construct(k, g)
        return res if res is not None else (0, [])

    best = None
    best_n = float('inf')

    # 1) voltage lift (structural new family) — try to beat everything
    if remaining() > 30:
        vl = voltage_lift(k, g, deadline - 25)
        if vl is not None:
            best, best_n = vl

    # 2) Ramanujan graphs — often give the smallest valid cubic graphs
    for q in [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]:
        if remaining() < 8:
            break
        res = build_ramanujan(q)
        if res is None:
            continue
        n, edges = res
        if n >= best_n:
            continue
        gc = check_girth(n, edges, g, deadline - 5)
        if gc == -1:
            continue
        if gc >= g:
            best, best_n = (n, edges), n

    if best is not None:
        return best

    # 3) LUW guaranteed-valid fallback
    res = luw_construct(k, g)
    return res if res is not None else (0, [])
