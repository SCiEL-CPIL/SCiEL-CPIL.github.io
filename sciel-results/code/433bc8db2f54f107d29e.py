def construct(k, g):
    import time
    import random
    from collections import deque

    start_time = time.time()
    deadline = start_time + 135.0

    # ---------- Known optimal / near-optimal cages ----------
    known_cages = {}

    # (3, 5) Petersen
    known_cages[(3, 5)] = (10, [(0,1),(1,2),(2,3),(3,4),(4,0),
                                (5,7),(7,9),(9,6),(6,8),(8,5),
                                (0,5),(1,6),(2,7),(3,8),(4,9)])
    # (3, 7) Heawood
    heawood_edges = []
    fano_lines = [(0,1,2),(0,3,4),(0,5,6),(1,3,5),(1,4,6),(2,3,6),(2,4,5)]
    for i, line in enumerate(fano_lines):
        for p in line:
            heawood_edges.append((p, 7 + i))
    known_cages[(3, 7)] = (14, heawood_edges)
    # (3, 3) triangle
    known_cages[(3, 3)] = (4, [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)])
    # (4, 3) K5
    known_cages[(4, 3)] = (5, [(i,j) for i in range(5) for j in range(i+1,5)])
    # (4, 4) K_{4,4}
    known_cages[(4, 4)] = (8, [(i,j) for i in range(4) for j in range(4,8)])
    # (5, 3) K6
    known_cages[(5, 3)] = (6, [(i,j) for i in range(6) for j in range(i+1,6)])
    # (5, 4) K_{5,5}
    known_cages[(5, 4)] = (10, [(i,j) for i in range(5) for j in range(5,10)])
    # (3, 4) K_{3,3}
    known_cages[(3, 4)] = (6, [(i,j) for i in range(3) for j in range(3,6)])

    if (k, g) in known_cages:
        return known_cages[(k, g)]

    # ---------- Girth verification (BFS with parent, fast) ----------
    def girth_ok(n, adj, g):
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if dist[u] + dist[v] + 1 < g:
                            return False
        return True

    def build_adj(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    # ---------- LUW construction ----------
    def build_luw(d, q):
        if q == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q

        rule = {}
        rule[2] = (1, 1)
        rule[3] = (2, 1)
        rule[4] = (1, 2)
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield (1 - side), tuple(y)

        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
            if time.time() > deadline:
                break
        return len(order), edges

    # ---------- Random k-regular bipartite graph (configuration model) ----------
    def build_random_bipartite(n_half, k, seed_val):
        rng = random.Random(seed_val)
        n = 2 * n_half
        left_stubs = [(i, j) for i in range(n_half) for j in range(k)]
        right_stubs = [(i, j) for i in range(n_half) for j in range(k)]
        rng.shuffle(right_stubs)
        edges = []
        edge_set = set()
        for (l, _), (r, _) in zip(left_stubs, right_stubs):
            u = l
            v = n_half + r
            e = (min(u, v), max(u, v))
            if e not in edge_set:
                edge_set.add(e)
                edges.append(e)
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if any(d != k for d in deg):
            return None
        adj = build_adj(n, edges)
        if not girth_ok(n, adj, g):
            return None
        return n, edges

    # ---------- Cayley graph on Z_p^d (for even k) ----------
    def build_cayley(p, d, k, seed_val):
        rng = random.Random(seed_val)
        n = p ** d
        if n < k + 1:
            return None

        def enc(elem):
            val = 0
            mult = 1
            for i in range(d):
                val += elem[i] * mult
                mult *= p
            return val

        # All nonzero elements
        all_nonzero = []
        for val in range(1, n):
            elem = []
            tmp = val
            for i in range(d):
                elem.append(tmp % p)
                tmp //= p
            all_nonzero.append(tuple(elem))

        if len(all_nonzero) < k:
            return None

        for attempt in range(20):
            if time.time() > deadline:
                return None
            half_k = k // 2
            S_half = rng.sample(all_nonzero, min(half_k, len(all_nonzero)))
            S = list(S_half)
            for s in S_half:
                inv = tuple((-s[i]) % p for i in range(d))
                if inv not in S:
                    S.append(inv)
            while len(S) > k:
                S.pop()
            while len(S) < k:
                remaining = [e for e in all_nonzero
                             if e not in S and tuple((-e[i]) % p for i in range(d)) not in S]
                if not remaining:
                    break
                s = rng.choice(remaining)
                S.append(s)
                inv = tuple((-s[i]) % p for i in range(d))
                if inv not in S:
                    S.append(inv)
            if len(S) != k:
                continue

            S_enc = [enc(s) for s in S]
            adj = [[] for _ in range(n)]
            for u in range(n):
                eu = []
                tmp = u
                for i in range(d):
                    eu.append(tmp % p)
                    tmp //= p
                for s in S_enc:
                    es = []
                    tmp = s
                    for i in range(d):
                        es.append(tmp % p)
                        tmp //= p
                    prod = tuple((eu[i] + es[i]) % p for i in range(d))
                    v = 0
                    mult = 1
                    for i in range(d):
                        v += prod[i] * mult
                        mult *= p
                    adj[u].append(v)

            if girth_ok(n, adj, g):
                edges = []
                edge_set = set()
                for u in range(n):
                    for v in adj[u]:
                        e = (min(u, v), max(u, v))
                        if e not in edge_set:
                            edge_set.add(e)
                            edges.append(e)
                return n, edges
        return None

    # ---------- Main search ----------
    best_n = None
    best_edges = None

    # Strategy 1: LUW (fast, algebraic, good girth)
    q = k
    if g % 2 == 1:
        start_d = g - 5
        if start_d < 3:
            start_d = 3
        if start_d % 2 == 0:
            start_d += 1
    else:
        start_d = g - 5
        if start_d < 2:
            start_d = 2
        if start_d % 2 == 1:
            start_d += 1

    for d in range(start_d, g + 3):
        if time.time() > deadline:
            break
        if g % 2 == 1 and d % 2 == 0:
            continue
        if g % 2 == 0 and d % 2 == 1:
            continue
        n, edges = build_luw(d, q)
        adj = build_adj(n, edges)
        if girth_ok(n, adj, g):
            best_n, best_edges = n, edges
            break

    # Strategy 2: Cayley graphs for even k
    if k % 2 == 0 and (best_n is None or best_n > 500):
        for p in [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
            if time.time() > deadline:
                break
            for d in range(1, 5):
                n = p ** d
                if n < k + 1 or n > 5000:
                    continue
                if (p - 1) % k != 0 and k > 1:
                    continue
                result = build_cayley(p, d, k, 42 + p * 100 + d)
                if result is not None:
                    n, edges = result
                    if best_n is None or n < best_n:
                        best_n, best_edges = n, edges
                if time.time() > deadline:
                    break

    # Strategy 3: Random regular bipartite graphs
    if best_n is None or best_n > 1000:
        if g % 2 == 1:
            depth = (g - 3) // 2
        else:
            depth = (g - 4) // 2
        min_n_half = 1
        for i in range(1, depth + 1):
            if i == 1:
                min_n_half += k
            else:
                min_n_half += k * (k - 1) ** (i - 1)

        for n_half in [min_n_half, min_n_half + k, min_n_half + 2 * k, min_n_half + 3 * k]:
            if time.time() > deadline:
                break
            for seed_val in range(50):
                if time.time() > deadline:
                    break
                result = build_random_bipartite(n_half, k, seed_val + 1000)
                if result is not None:
                    n, edges = result
                    if best_n is None or n < best_n:
                        best_n, best_edges = n, edges
                    break

    if best_n is None:
        # Fallback: LUW with largest d
        d = g - 2
        if d < 3:
            d = 3
        n, edges = build_luw(d, q)
        best_n, best_edges = n, edges

    return best_n, best_edges
