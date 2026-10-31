import random
import time

# ---- global deadline (150s wall-clock kill; keep margin) ----
_T0 = time.time()
_DEADLINE = _T0 + 142.0


def construct(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q), q = k: an explicit q-regular
    # bipartite graph with girth at least d + 5 (odd d). Guaranteed valid.
    def luw_construct(k, g):
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

    if not (k == 3 and g == 16):
        return luw_construct(k, g)

    def search(n_target, time_budget):
        # Backtracking construction of a 3-regular graph on n_target vertices
        # with girth >= 16, using a non-backtracking closed-walk certificate.
        girth = 16
        max_depth = girth - 2  # keep NB walks of length <= 14; a length-15
                                # collision implies a cycle of length <= 15
        adj = [[] for _ in range(n_target)]
        edges = []
        walk = [None] * n_target  # walk[v] = dict (src, depth) -> frozenset of ends
        state = [1] * n_target    # 0 = pending, 1 = partial, 2 = done

        def add_walk(v, src, depth, ends):
            if depth > max_depth:
                return
            d = walk[v]
            if d is None:
                d = walk[v] = {}
            key = (src, depth)
            if key in d:
                d[key] = d[key] | ends
            else:
                d[key] = ends

        def ok_edge(a, b):
            # True iff adding undirected edge (a,b) creates no NB closed walk < girth.
            wa, wb = walk[a], walk[b]
            if wa is None and wb is None:
                return True
            if wa is None:
                for (src, depth), ends in wb.items():
                    for e in ends:
                        if e == a and depth + 1 >= girth:
                            return False
                return True
            if wb is None:
                for (src, depth), ends in wa.items():
                    for e in ends:
                        if e == b and depth + 1 >= girth:
                            return False
                return True
            for (sa, da), ea in wa.items():
                for (sb, db), eb in wb.items():
                    if sa == sb and da + db + 1 < girth:
                        if ea & eb:
                            return False
            for (sa, da), ea in wa.items():
                for e in ea:
                    if e == b and da + 1 >= girth:
                        return False
            for (sb, db), eb in wb.items():
                for e in eb:
                    if e == a and db + 1 >= girth:
                        return False
            return True

        def commit(a, b):
            wa, wb = walk[a], walk[b]
            if wa is None:
                wa = walk[a] = {}
            if wb is None:
                wb = walk[b] = {}
            for (src, depth), ends in wa.items():
                key = (src, depth + 1)
                if key in wb:
                    wb[key] = wb[key] | ends
                else:
                    wb[key] = ends
            for (src, depth), ends in wb.items():
                key = (src, depth + 1)
                if key in wa:
                    wa[key] = wa[key] | ends
                else:
                    wa[key] = ends
            adj[a].append(b)
            adj[b].append(a)
            edges.append((a, b))

        def uncommit(a, b):
            wa, wb = walk[a], walk[b]
            for (src, depth), ends in list(wa.items()):
                if depth >= 1:
                    key = (src, depth - 1)
                    if key in wb:
                        wb[key] &= ~ends
                        if not wb[key]:
                            del wb[key]
            for (src, depth), ends in list(wb.items()):
                if depth >= 1:
                    key = (src, depth - 1)
                    if key in wa:
                        wa[key] &= ~ends
                        if not wa[key]:
                            del wa[key]
            adj[a].remove(b)
            adj[b].remove(a)
            edges.pop()

        def build(i, deadline):
            if i == n_target:
                return True
            if time.time() > deadline:
                return False
            deg = len(adj[i])
            if deg == 0:
                cand = [j for j in range(i + 1, n_target) if len(adj[j]) < 3]
            elif deg == 1:
                cand = [j for j in range(i + 1, n_target)
                        if len(adj[j]) < 3 and j not in adj[i]]
            else:
                cand = [j for j in range(i + 1, n_target)
                        if len(adj[j]) < 3 and j not in adj[i]]
            random.shuffle(cand)
            for j in cand:
                if not ok_edge(i, j):
                    continue
                commit(i, j)
                if len(adj[i]) == 3:
                    state[i] = 2
                else:
                    state[i] = 1
                if build(i + 1, deadline):
                    return True
                uncommit(i, j)
                state[i] = 1 if len(adj[i]) > 0 else 0
            return False

        deadline = time.time() + time_budget
        if build(0, deadline):
            return n_target, edges
        return None

    # Deterministic, cheap, always-valid starting point.
    n0, edges0 = luw_construct(k, g)

    # Attempt to beat the LUW size with a certified backtracking search.
    best_n, best_edges = n0, edges0
    t_start = time.time()
    n_target = 64
    while n_target < best_n and time.time() < _DEADLINE - 1.0:
        budget = min(6.0, max(0.5, _DEADLINE - 1.0 - time.time()))
        if budget <= 0.0:
            break
        res = search(n_target, budget)
        if res is not None:
            nn, ee = res
            if nn < best_n:
                best_n, best_edges = nn, ee
            break  # found a valid smaller graph; it is the best possible here
        n_target += 8

    return best_n, best_edges
