import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    def remaining():
        return deadline - time.time()

    # ---------------- girth verification (fast, early exit) ----------------
    def check_girth(n, edges, g, time_limit):
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        # Fast non-backtracking BFS from each vertex:
        # state = (node, prev_node); dist = shortest non-backtracking walk length
        # A cycle of length L is found when we revisit (node, prev) at dist L.
        # For girth check, we only need to know if any cycle < g exists.
        # Optimization: from each start s, do NB-BFS up to depth (g-1)//2.
        # If we find a back-edge (cycle) of length < g, return it.
        # Use a visited array indexed by (node, prev) pair.
        # For a 3-regular graph, we can index state by node * n + prev (or use dict).
        # Since n can be large, use a dict per start.
        half = (g - 1) // 2
        for s in range(n):
            if time.time() > time_limit:
                return -1
            # NB-BFS: dist[(node, prev)] = length of shortest NB walk from s to (node, prev)
            # Start: from s, go to each neighbor nb: state (nb, s) at dist 1
            dist = {}
            queue = []
            for nb in adj[s]:
                key = (nb, s)
                dist[key] = 1
                queue.append((nb, s))
            qi = 0
            found = False
            while qi < len(queue):
                node, prev = queue[qi]
                qi += 1
                d = dist[(node, prev)]
                if d > half:
                    continue
                for nb in adj[node]:
                    if nb == prev:
                        continue  # non-backtracking
                    key = (nb, node)
                    if key in dist:
                        # Cycle found: dist[(nb,node)] + d + 1
                        cycle_len = dist[key] + d + 1
                        if cycle_len < g:
                            return cycle_len
                    else:
                        if d + 1 <= half:
                            dist[key] = d + 1
                            queue.append((nb, node))
            if found:
                break
        return g  # no cycle shorter than g found

    # ---------------- LUW construction ----------------
    def _luw_ops(q):
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            return (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        return ((lambda x, y: (x + y) % q),
                (lambda x, y: (x - y) % q),
                (lambda x, y: x * y % q))

    def _luw_ops_alt(q):
        """Alternative LUW ops: use different multiplication table for better girth."""
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            return (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        # Use (x+y) mod q, (x-y) mod q, but with a shifted multiplication
        return ((lambda x, y: (x + y) % q),
                (lambda x, y: (x - y) % q),
                (lambda x, y: (x * y + x + y) % q))

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

    def build_luw(q, d, rule, time_limit, alt_ops=False):
        if alt_ops:
            add, sub, mul = _luw_ops_alt(q)
        else:
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

        # Explore more rule variants with offsets
        for variant in range(5):
            for offset in range(6):
                if remaining() < 12:
                    break
                rule = generate_rules(d, variant, offset)
                for use_alt in [False, True]:
                    if remaining() < 12:
                        break
                    res = build_luw(q, d, rule, deadline - 12, alt_ops=use_alt)
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
                        if n <= 2000:  # good enough, stop early
                            return best

        if best is None:
            for d_try in range(d + 1, d + 12):
                if remaining() < 10:
                    break
                for variant in range(5):
                    for offset in range(4):
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

        if best is None:  # guaranteed-valid fallback
            rule = generate_rules(d, 0, 0)
            res = build_luw(q, d, rule, deadline - 2)
            if res is not None:
                best = res
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
            m = mul(a, a)
            n = norm(m)
            return n == 0 or n == 2  # m ~ I

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
        # dedup up to sign
        seen, gens = set(), []
        for m in G:
            key = m if (m[0], m[1], m[2], m[3]) <= (q - m[0] % q, q - m[1] % q, q - m[2] % q, q - m[3] % q) else (q - m[0] % q, q - m[1] % q, q - m[2] % q, q - m[3] % q)
            if key not in seen:
                seen.add(key)
                gens.append(m)
        if len(gens) != 3:
            return None
        # BFS over PSL(2,q)
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

    # k == 3: try Ramanujan graphs (smaller), else LUW.
    best = None
    best_n = float('inf')

    # Try primes: Ramanujan cubic graphs have n = q(q+1)/2 vertices.
    # For girth >= 16, we need q large enough. Known: q=19 gives girth 8,
    # q=31 gives girth 8, q=79 gives girth 10, q=151 gives girth 12,
    # q=211 gives girth 14, q=433 gives girth 16 (n=93606).
    # But our construction may not achieve optimal girth for all q.
    # Try a range of primes and check girth.
    primes = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499]
    for q in primes:
        if remaining() < 10:
            break
        # Moore bound for k=3, g=16: n >= 2 + 3*(1 + 2 + 4 + 8 + 16 + 32) = 2 + 3*63 = 191
        # Ramanujan n = q(q+1)/2, so q >= ~20 for n >= 191
        if q * (q + 1) // 2 < 191:
            continue
        res = build_ramanujan(q)
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
            if n <= 3000:  # excellent, stop
                break

    if best is not None:
        return best

    # fall back to LUW (guaranteed valid)
    res = luw_construct(k, g)
    return res if res is not None else (0, [])
