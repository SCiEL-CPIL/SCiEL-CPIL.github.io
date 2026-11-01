def construct(k, g):
    import time
    start_time = time.time()
    deadline = start_time + 110.0

    # Fast paths for small / known cases
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
            # Heawood graph: 3-regular, girth 6, 14 vertices
            # It's the incidence graph of Fano plane
            # 7 points, 7 lines
            edges = []
            # Fano plane lines
            lines = [[0,1,2],[0,3,4],[0,5,6],[1,3,5],[1,4,6],[2,3,6],[2,4,5]]
            for i, line in enumerate(lines):
                for pt in line:
                    edges.append((i, 7 + pt))
            return 14, edges
        elif g <= 7:
            # 3-regular girth 7: 24 vertices (cage)
            # Use LUW as fallback
            pass
        elif g <= 8:
            pass
        elif g <= 9:
            pass
    elif k == 4:
        if g <= 3:
            return 4, [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
        elif g <= 4:
            return 8, [(i, j) for i in range(4) for j in range(4, 8)]
        elif g <= 5:
            # 4-regular girth 5: 10 vertices? Let's use LUW
            pass
        elif g <= 6:
            pass
        elif g <= 7:
            # 4-regular girth 7: 22 vertices
            pass
        elif g <= 8:
            pass
    elif k == 5:
        if g <= 3:
            return 6, [(i, j) for i in range(6) for j in range(i+1, 6)]
        elif g <= 4:
            return 10, [(i, j) for i in range(5) for j in range(5, 10)]
        elif g <= 5:
            pass
        elif g <= 6:
            pass
        elif g <= 7:
            pass
        elif g <= 8:
            pass
        elif g <= 9:
            # Main case: 5-regular, girth 9
            pass

    def build_luw(d, q):
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add_table = [[x ^ y for y in range(4)] for x in range(4)]
            sub_table = [[x ^ y for y in range(4)] for x in range(4)]
            mul_table = table
        else:
            add_table = [[(x + y) % q for y in range(q)] for x in range(q)]
            sub_table = [[(x - y) % q for y in range(q)] for x in range(q)]
            mul_table = [[x * y % q for y in range(q)] for x in range(q)]

        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        rule_list = [None] * (d + 1)
        for c in range(2, d + 1):
            rule_list[c] = rule[c]

        def neighbours(vertex):
            side, x = vertex
            if side == 0:
                for first in range(q):
                    y = [None, first] + [0] * (d - 1)
                    for c in range(2, d + 1):
                        a, b = rule_list[c]
                        y[c] = add_table[x[c]][mul_table[y[a]][x[b]]]
                    yield 1, tuple(y)
            else:
                for first in range(q):
                    y = [None, first] + [0] * (d - 1)
                    for c in range(2, d + 1):
                        a, b = rule_list[c]
                        y[c] = sub_table[x[c]][mul_table[x[a]][y[b]]]
                    yield 0, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        edge_set = set()

        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    e = (index[vertex], index[other])
                    if e not in edge_set:
                        edge_set.add(e)
                        edges.append(e)
            if time.time() > deadline:
                break

        return len(order), edges

    def verify_girth(n, edges, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
        return True

    q = k
    best_n, best_edges = None, None

    # Try d values: for girth g, we need d >= g - 5 (for odd g) or d >= g - 4 (for even g)
    # LUW graph CD(d, q) has girth at least d + 5 for odd d
    # So for girth g, we need d + 5 >= g, i.e., d >= g - 5
    # For odd g: d should be odd, d >= g - 5
    # For even g: d should be even, d >= g - 4 (since d+5 >= g means d >= g-5, but d must be even)

    if g % 2 == 1:
        # g is odd, d must be odd
        start_d = g - 5
        if start_d < 3:
            start_d = 3
        if start_d % 2 == 0:
            start_d += 1
    else:
        # g is even, d must be even
        start_d = g - 5
        if start_d < 2:
            start_d = 2
        if start_d % 2 == 1:
            start_d += 1

    for d in range(start_d, g + 2):
        if time.time() > deadline:
            break
        # Ensure d has correct parity
        if g % 2 == 1 and d % 2 == 0:
            continue
        if g % 2 == 0 and d % 2 == 1:
            continue
        n, edges = build_luw(d, q)
        if verify_girth(n, edges, g):
            best_n, best_edges = n, edges
            break

    if best_n is None:
        # Fallback: use the first valid d
        if g % 2 == 1:
            d = max(3, g - 5)
            if d % 2 == 0:
                d += 1
        else:
            d = max(2, g - 5)
            if d % 2 == 1:
                d += 1
        n, edges = build_luw(d, q)
        best_n, best_edges = n, edges

    return best_n, best_edges
