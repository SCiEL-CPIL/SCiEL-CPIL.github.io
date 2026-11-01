"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(k, g):
    import time
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin

    def verify_graph(n, edges, k, g):
        """Verify: simple, k-regular, girth >= g. Returns (n, edges) if valid, else None."""
        if time.time() > deadline:
            return None
        # Check simplicity and degree
        edge_set = set()
        degree = [0] * n
        for u, v in edges:
            if u == v:
                return None
            if u < 0 or u >= n or v < 0 or v >= n:
                return None
            key = (min(u, v), max(u, v))
            if key in edge_set:
                return None
            edge_set.add(key)
            degree[u] += 1
            degree[v] += 1
        for i in range(n):
            if degree[i] != k:
                return None
        # Check girth >= g using BFS from each vertex
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        # For girth >= g, no cycle of length < g
        # BFS from each vertex, if we find a back edge creating cycle < g, fail
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            qi = 0
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                if dist[node] >= (g - 1) // 2 + 1:
                    continue
                for nb in adj[node]:
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
                    elif nb != parent[node]:
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < g:
                            return None
        return (n, edges)

    def build_luw(k, g):
        """LUW construction fallback."""
        q = k
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)

        def build_luw_d(d):
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

        target_girth_check = g if g % 2 == 0 else g + 1
        min_d = max(2, (target_girth_check - 2) // 2)
        for d in range(min_d, min_d + 15):
            if time.time() > deadline:
                break
            try:
                n, edges = build_luw_d(d)
                result = verify_graph(n, edges, k, g)
                if result is not None:
                    return result
            except Exception:
                continue
        return None

    # Special case: k=5, g<=10 -> use GQ(2,4) construction for 50 vertices, girth 10
    if k == 5 and g <= 10:
        # GQ(2,4): 5-regular bipartite graph on 50 vertices with girth 10
        # Points: 25 points in PG(3,4) -> use affine 3-space over F4 plus infinity
        # Actually, GQ(2,4) has (s+1)(s(t+1)+1) = 3*(2*5+1) = 39 points? No.
        # GQ(s,t) has (s+1)(st+s+t+1) points and (t+1)(s(t+1)+1) lines.
        # GQ(2,4): points = 3*(8+2+4+1) = 3*15 = 45? No.
        # GQ(s,t): number of points = (s+1)(st+s+t+1)
        # GQ(2,4): 3*(8+2+4+1) = 3*15 = 45 points, 30 lines? 
        # Each line has s+1=3 points, each point on t+1=5 lines.
        # Incidence graph: 45+30=75 vertices, 5-regular? Each point connects to 5 lines, each line to 3 points.
        # That's not 5-regular on both sides.
        
        # Let me use a different known construction.
        # The (5,9)-cage: known constructions exist.
        # For k=5, g=9: the smallest known is large. But we just need ANY valid graph.
        # 
        # Alternative: Use a circulant or known small graph.
        # Actually, let's use the LUW construction which is known to work.
        pass

    # Use LUW construction
    result = build_luw(k, g)
    if result is not None:
        return result

    # Absolute fallback: should not reach here
    raise RuntimeError("No valid graph found")
# EVOLVE-BLOCK-END
