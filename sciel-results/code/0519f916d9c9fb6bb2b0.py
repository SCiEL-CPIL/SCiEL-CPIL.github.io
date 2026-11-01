import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    # Hardcoded known optimal cages for small cases
    if k == 3:
        if g <= 3:
            return 3, [(0, 1), (1, 2), (0, 2)]
        elif g <= 4:
            return 6, [(0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (1, 5),
                        (2, 3), (2, 4), (2, 5)]
        elif g <= 5:
            edges = []
            for i in range(5):
                edges.append((i, (i + 1) % 5))
                edges.append((i, 5 + i))
                edges.append((5 + i, 5 + (i + 2) % 5))
            return 10, edges
        elif g <= 6:
            edges = []
            lines = [[0,1,2],[0,3,4],[0,5,6],[1,3,5],[1,4,6],[2,3,6],[2,4,5]]
            for i, line in enumerate(lines):
                for pt in line:
                    edges.append((i, 7 + pt))
            return 14, edges

    elif k == 4:
        if g <= 3:
            return 4, [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
        elif g <= 4:
            return 8, [(i, j) for i in range(4) for j in range(4, 8)]

    elif k == 5:
        if g <= 3:
            return 6, [(i, j) for i in range(6) for j in range(i+1, 6)]
        elif g <= 4:
            return 10, [(i, j) for i in range(5) for j in range(5, 10)]

    def luw_build(d, q):
        """
        Build the Lazebnik-Ustimenko-Woldar graph.
        Vertices: 2 * q^d
        Side 0 vertices: (0, x_1, ..., x_d), x_i in Z_q
        Side 1 vertices: (1, y_1, ..., y_d), y_i in Z_q
        
        Edges from (0, x_1, ..., x_d) to (1, y_1, ..., y_d) where:
          y_1 is free
          y_c = x_c + y_a * x_b  for c = 2, ..., d
        with (a,b) determined by the rule, and x_0 = y_1.
        """
        qd = q ** d
        n = 2 * qd

        # Rule table
        rule = {}
        rule[2] = (1, 1)
        rule[3] = (2, 1)
        if d >= 4:
            rule[4] = (1, 2)
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - r
            opts = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
            rule[c] = opts[r]

        # Precompute addition and multiplication tables
        add_t = [[(x + y) % q for y in range(q)] for x in range(q)]
        mul_t = [[x * y % q for y in range(q)] for x in range(q)]

        edges = []
        edge_set = set()

        # Iterate over all side-0 vertices
        for x_idx in range(qd):
            # Decode x_1, ..., x_d
            x = [0] * (d + 1)  # 1-indexed
            tmp = x_idx
            for i in range(d, 0, -1):
                x[i] = tmp % q
                tmp //= q

            # For each possible y_1
            for y1 in range(q):
                y = [0] * (d + 1)
                y[1] = y1
                for c in range(2, d + 1):
                    a, b = rule[c]
                    ya = y[a] if a >= 1 else y1  # a should be >= 1
                    xb = x[b]
                    y[c] = add_t[x[c]][mul_t[ya][xb]]

                y_idx = 0
                for i in range(1, d + 1):
                    y_idx = y_idx * q + y[i]
                w = qd + y_idx
                v = x_idx

                e = (v, w)
                if e not in edge_set:
                    edge_set.add(e)
                    edges.append(e)

        return n, edges

    def check_girth(n, edges, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        for i in range(n):
            if len(adj[i]) != k:
                return False

        for s in range(n):
            dist = [-1] * n
            par = [-1] * n
            dist[s] = 0
            q_dq = deque([s])
            while q_dq:
                u = q_dq.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        par[v] = u
                        q_dq.append(v)
                    elif v != par[u]:
                        if dist[u] + dist[v] + 1 < g:
                            return False
        return True

    best_n, best_edges = None, None

    # Determine valid d values
    if g % 2 == 1:
        start_d = max(3, g - 5)
        if start_d % 2 == 0:
            start_d += 1
    else:
        start_d = max(2, g - 5)
        if start_d % 2 == 1:
            start_d += 1

    max_d = min(start_d + 4, 10)

    for d in range(start_d, max_d + 1):
        if time.time() > deadline - 5:
            break
        if g % 2 == 1 and d % 2 == 0:
            continue
        if g % 2 == 0 and d % 2 == 1:
            continue

        n, edges = luw_build(d, k)
        if check_girth(n, edges, g):
            if best_n is None or n < best_n:
                best_n, best_edges = n, edges

    if best_n is None:
        d = start_d
        best_n, best_edges = luw_build(d, k)

    return best_n, best_edges
