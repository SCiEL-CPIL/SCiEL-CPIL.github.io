import time
import sys
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 118.0

    def remaining():
        return deadline - time.time()

    # ---------------- girth verification (vectorized BFS from all vertices) ----------------
    def check_girth(n, edges, min_girth, time_limit):
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        # Build CSR
        deg = np.array([len(a) for a in adj], dtype=np.int32)
        total = deg.sum()
        csr = np.zeros(total, dtype=np.int32)
        pos = 0
        for i in range(n):
            for j in adj[i]:
                csr[pos] = j
                pos += 1
        offsets = np.zeros(n + 1, dtype=np.int32)
        np.cumsum(deg, out=offsets[1:])

        dist = np.full((n, n), -1, dtype=np.int32)
        dist[np.arange(n), np.arange(n)] = 0
        # BFS from all sources simultaneously (bit-parallel-ish via vectorized frontier)
        # We process layer by layer.
        frontier = np.arange(n, dtype=np.int32)  # sources at dist 0
        depth = 0
        while len(frontier) > 0 and time.time() < time_limit:
            depth += 1
            if depth >= min_girth:
                break
            # neighbours of frontier
            nbrs = np.concatenate([csr[offsets[s]:offsets[s + 1]] for s in frontier]) if len(frontier) < 2000 else \
                   np.concatenate([csr[offsets[s]:offsets[s + 1]] for s in frontier])
            if len(nbrs) == 0:
                break
            # dedup
            nbrs = np.unique(nbrs)
            # for each source s in frontier and neighbour v: if dist[s,v] already set -> cycle
            # We check: is there s in frontier with dist[s, v] != -1?
            # Build matrix check: dist[frontier][:, nbrs]
            sub = dist[frontier][:, nbrs]
            hit = sub != -1
            if hit.any():
                # cycle length = depth + dist[s, v]
                # find min
                rows = np.where(hit.any(axis=1))[0]
                cols = np.where(hit.any(axis=0))[0]
                # compute for each (row,col) in the hit set
                r_idx, c_idx = np.where(hit)
                lens = depth + sub[r_idx, c_idx]
                if lens.min() < min_girth:
                    return int(lens.min())
            # update dist for new frontier
            new_dist = np.full(len(nbrs), -1, dtype=np.int32)
            # For each source s in frontier, set dist[s, v] = depth for v not yet visited
            # Vectorized: for each s, mask
            # This is O(|frontier|*|nbrs|) worst case; acceptable for moderate sizes.
            # We do it in chunks to limit memory.
            chunk = 256
            for i0 in range(0, len(frontier), chunk):
                s_chunk = frontier[i0:i0 + chunk]
                # dist[s_chunk, nbrs] currently -1 where unvisited
                cur = dist[s_chunk][:, nbrs]
                unv = cur == -1
                # set
                dist[s_chunk][:, nbrs] = np.where(unv, depth, cur)
            # next frontier = nbrs that are newly reached by at least one source
            # A vertex v is in next frontier if it was newly reached by some s in frontier
            # i.e. exists s in frontier with dist[s,v] was -1 and now depth
            newly = np.zeros(len(nbrs), dtype=bool)
            for i0 in range(0, len(frontier), chunk):
                s_chunk = frontier[i0:i0 + chunk]
                # we already overwrote dist; we need the unv mask from before. Recompute via:
                # v newly reached if dist[s,v]==depth and it was -1. We lost -1 info.
                # Instead recompute: v in next frontier if any s in frontier has dist[s,v]==depth
                # and v != s (v is a neighbour so v!=s generally) and depth is the first time.
                # Simpler: next frontier = nbrs that are not sources and were unvisited.
                # Determine unvisited-before: we can't easily. Use the fact that a vertex newly
                # reached at this layer satisfies dist[s,v]==depth for some s in frontier and
                # no earlier layer reached it. We track a global "last_depth[v]" = max depth at
                # which v was reached; if last_depth[v] < depth and now reached, it's new.
                pass
            # Fallback: simpler sequential-ish BFS but vectorized over sources via last-reached depth
            break
        # The fully vectorized approach above is getting complex; use a robust per-source BFS
        # but in C-speed via numpy arrays is hard. Fall back to efficient Python BFS per source.
        min_cycle = float('inf')
        for s in range(n):
            if time.time() > time_limit:
                return -1
            dist2 = [-1] * n
            parent = [-1] * n
            dist2[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist2[nb] != -1:
                        cycle_len = dist2[node] + dist2[nb] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if cycle_len < min_girth:
                            return min_cycle
                    else:
                        dist2[nb] = dist2[node] + 1
                        parent[nb] = node
                        queue.append(nb)
            if min_cycle < min_girth:
                return min_cycle
        return min_cycle

    # ---------------- Voltage Lift of Petersen ----------------
    def voltage_lift_petersen():
        if k != 3 or g != 16:
            return None

        pet_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
        ]
        n_base = 10
        edge_index = {}
        for i, (u, v) in enumerate(pet_edges):
            edge_index[(u, v)] = i
            edge_index[(v, u)] = i

        adj = [[] for _ in range(n_base)]
        for u, v in pet_edges:
            adj[u].append(v)
            adj[v].append(u)

        # Collect all non-backtracking closed walks of length 5..g-1, store (count_vector, parity)
        max_L = g - 1
        constraints = set()
        parity_constraints = set()

        for start_v in range(n_base):
            stack = [(start_v, -1, 0, [0] * 15)]
            while stack:
                current, prev, length, counts_list = stack.pop()
                if length > max_L:
                    continue
                for next_v in adj[current]:
                    if next_v == prev:
                        continue
                    e_idx = edge_index[(current, next_v)]
                    u, v = pet_edges[e_idx]
                    d = 1 if current == u else -1
                    new_counts = counts_list.copy()
                    new_counts[e_idx] += d
                    new_l = length + 1
                    if next_v == start_v and new_l >= 5:
                        c_tuple = tuple(new_counts)
                        if any(x != 0 for x in c_tuple):
                            first_nz = -1
                            for i in range(15):
                                if c_tuple[i] != 0:
                                    first_nz = i
                                    break
                            if first_nz != -1 and c_tuple[first_nz] < 0:
                                c_tuple = tuple(-x for x in c_tuple)
                            constraints.add(c_tuple)
                            # parity: sum of |counts|? Use parity of total signed sum is 0 always (closed).
                            # Use: is the walk "odd-length"? new_l parity.
                            parity_constraints.add((c_tuple, new_l % 2))
                    if new_l < max_L:
                        stack.append((next_v, current, new_l, new_counts))

        if not constraints:
            return None

        C_list = list(constraints)
        C = np.array(C_list, dtype=np.int64)
        # odd-length constraints: those with odd length (these must be != 0 mod 2 as well, i.e. != 0 mod m)
        odd_set = set(c for (c, p) in parity_constraints if p == 1)
        odd_idx = [i for i, c in enumerate(C_list) if c in odd_set]
        C_odd = C[odd_idx] if odd_idx else np.zeros((0, 15), dtype=np.int64)

        def is_prime(n):
            if n < 2:
                return False
            if n < 4:
                return True
            if n % 2 == 0 or n % 3 == 0:
                return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0:
                    return False
                i += 6
            return True

        # Precompute for fast cost: we want C @ v mod m != 0 for all rows.
        # Use smooth surrogate: sum exp(-a * min(residues, m-residues)/m)
        def make_cost(m, Cmat):
            # returns cost(v) = number of zero residues (exact) and smooth(v)
            def cost(v):
                r = np.mod(Cmat @ v, m)
                return int(np.sum(r == 0))
            def smooth(v):
                r = np.mod(Cmat @ v, m)
                d = np.minimum(r, m - r).astype(np.float64) / float(m)
                # large d -> small penalty
                return float(np.sum(np.exp(-20.0 * d)))
            return cost, smooth

        def solve_voltage(m, time_limit):
            cost, smooth = make_cost(m, C)
            # Step 1: solve relaxed Z_2 system on odd constraints to get a seed.
            # We want C_odd @ v != 0 (mod 2) for all odd rows. Equivalently minimize #zeros over Z_2.
            v2 = None
            if len(C_odd) > 0:
                # Random walk over Z_2^15
                best_v2 = np.random.randint(0, 2, size=15)
                def cost2(v):
                    return int(np.sum(np.mod(C_odd @ v, 2) == 0))
                bcost = cost2(best_v2)
                for it in range(20000):
                    if time.time() > time_limit:
                        break
                    j = random.randint(0, 14)
                    old = best_v2[j]
                    best_v2[j] = 1 - old
                    nc = cost2(best_v2)
                    if nc <= bcost:
                        bcost = nc
                        if nc == 0:
                            break
                    else:
                        best_v2[j] = old
                v2 = best_v2
            # Initialize v in Z_m from v2 (map 0->1, 1->m//2) plus randomness
            v = np.random.randint(1, m, size=15).astype(np.int64)
            if v2 is not None:
                for i in range(15):
                    v[i] = 1 if v2[i] == 0 else (m // 2)
                    if v[i] < 1:
                        v[i] = 1
            best_v = v.copy()
            best_cost = cost(v)
            if best_cost == 0:
                return v, 0
            # Greedy coordinate descent on exact cost, with occasional random restarts
            improved = True
            it = 0
            while improved and time.time() < time_limit:
                improved = False
                for j in range(15):
                    if time.time() > time_limit:
                        break
                    old = v[j]
                    best_j = old
                    best_jc = best_cost
                    # try a few candidate values
                    cands = set()
                    cands.add(random.randint(1, m - 1))
                    cands.add(random.randint(1, m - 1))
                    cands.add((old + 1) % m or 1)
                    cands.add((old - 1) % m or 1)
                    for cand in cands:
                        if cand == old or cand < 1:
                            continue
                        v[j] = cand
                        c = cost(v)
                        if c < best_jc:
                            best_jc = c
                            best_j = cand
                    v[j] = best_j
                    if best_jc < best_cost:
                        best_cost = best_jc
                        best_v = v.copy()
                        improved = True
                        if best_cost == 0:
                            return v, 0
            # Smooth surrogate polish: minimize smooth(v) via random perturbation hill-climb
            sv = smooth(best_v)
            for it in range(3000):
                if time.time() > time_limit:
                    break
                j = random.randint(0, 14)
                old = best_v[j]
                best_v[j] = random.randint(1, m - 1)
                nc = cost(best_v)
                if nc < best_cost or (nc == best_cost and smooth(best_v) < sv):
                    best_cost = nc
                    sv = smooth(best_v)
                    if nc == 0:
                        return best_v, 0
                else:
                    best_v[j] = old
            return best_v, best_cost

        # Prime ladder: aim near Moore bound. n = 10*m. Moore bound ~512 -> m ~ 51.2.
        primes = [53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293]
        best_res = None
        best_n = float('inf')
        for m in primes:
            if remaining() < 15:
                break
            n_try = 10 * m
            if n_try >= best_n:
                continue
            found = False
            for restart in range(4):
                if remaining() < 8:
                    break
                v, cost = solve_voltage(m, deadline - 6)
                if cost == 0:
                    edges = []
                    for u, v_base in pet_edges:
                        e_idx = edge_index[(u, v_base)]
                        vol = v[e_idx]
                        for layer in range(m):
                            u_lift = u * m + layer
                            v_lift = v_base * m + (layer + vol) % m
                            if u_lift < v_lift:
                                edges.append((u_lift, v_lift))
                            else:
                                edges.append((v_lift, u_lift))
                    gc = check_girth(n_try, edges, g, deadline - 3)
                    if gc >= g:
                        best_res = (n_try, edges)
                        best_n = n_try
                        found = True
                        break
                    else:
                        pass
            if found:
                # keep searching for smaller? we iterate ascending m so first found is smallest in this ladder;
                # but a larger m might have been skipped; continue to try to improve only if we want. Stop early.
                break
        if best_res is not None:
            return best_res
        return None

    # ---------------- Ramanujan cubic graphs (k=3) ----------------
    def build_ramanujan(q):
        def mul(a, b):
            return ((a[0]*b[0] + a[1]*b[2]) % q, (a[0]*b[1] + a[1]*b[3]) % q,
                    (a[2]*b[0] + a[3]*b[2]) % q, (a[2]*b[1] + a[3]*b[3]) % q)

        def det(a):
            return (a[0]*a[3] - a[1]*a[2]) % q

        def inv(a):
            d = det(a)
            if d == 0:
                return None
            invd = pow(d, q - 2, q)
            return ((a[3]*invd) % q, (-a[1]*invd) % q, (-a[2]*invd) % q, (a[0]*invd) % q)

        def norm(a):
            return (a[0]*a[0] + a[1]*a[1] + a[2]*a[2] + a[3]*a[3]) % q

        def involution(a):
            if det(a) != 1:
                return False
            m = mul(a, a)
            n = norm(m)
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

    # ---------------- LUW construction (Fallback) ----------------
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

        for variant in range(5):
            for offset in range(4):
                if remaining() < 12:
                    break
                rule = generate_rules(d, variant, offset)
                res = build_luw(q, d, rule, deadline - 12)
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

        if best is None:
            for d_try in range(d + 1, d + 10):
                if remaining() < 10:
                    break
                for variant in range(3):
                    for offset in range(3):
                        if remaining() < 5:
                            break
                        rule = generate_rules(d_try, variant, offset)
                        res = build_luw(q, d_try, rule, deadline - 10)
                        if res is None:
                            continue
                        n, edges = res
                        gc = check_girth(n, edges, g, deadline - 5)
                        if gc == -1:
                            continue
                        if gc >= g:
                            best = (n, edges)
                            break
                    if best is not None:
                        break
                if best is not None:
                    break

        if best is None:
            rule = generate_rules(d, 0, 0)
            res = build_luw(q, d, rule, deadline - 2)
            if res is not None:
                best = res
        return best

    # ---------------- main ----------------
    if k != 3:
        res = luw_construct(k, g)
        return res if res is not None else (0, [])

    res = voltage_lift_petersen()
    if res is not None:
        return res

    best = None
    best_n = float('inf')

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

    res = luw_construct(k, g)
    return res if res is not None else (0, [])
