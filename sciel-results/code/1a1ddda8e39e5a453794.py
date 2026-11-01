"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(k, g):
    import random
    import math
    import time
    from collections import deque

    # Construct a 3-regular graph using a voltage lift of K_{3,3} over Z_m.
    # Base graph: K_{3,3} with bipartition A={0,1,2}, B={0,1,2} (reindexed).
    # Lifted graph has n = 6*m vertices.
    # Vertex encoding: side * 3 * m + base * m + shift, where side in {0,1}, base in {0,1,2}, shift in {0,...,m-1}.
    # Edge from (0, i, s) to (1, j, (s + v[i][j]) % m).

    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer

    # Moore bound for (3, g) cage
    if g % 2 == 0:
        moore_bound = 3 * (2 ** (g // 2 - 1) - 1)
    else:
        moore_bound = 3 * (2 ** ((g - 1) // 2 - 1) - 1) if (g - 1) // 2 >= 1 else 3

    m_min = max(1, math.ceil(moore_bound / 6))

    def build_adjacency(m, voltages):
        """Build adjacency list for the lifted graph."""
        n = 6 * m
        adj = [[] for _ in range(n)]
        for i in range(3):
            for j in range(3):
                v = voltages[i][j]
                for s in range(m):
                    u = i * m + s
                    w = 3 * m + j * m + (s + v) % m
                    adj[u].append(w)
                    adj[w].append(u)
        return adj

    def check_girth_fast(m, voltages, required_girth):
        """Check girth >= required_girth using BFS from each vertex with early termination."""
        n = 6 * m
        adj = build_adjacency(m, voltages)
        half_girth = required_girth // 2

        for start in range(n):
            # BFS from start
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])

            while queue:
                u = queue.popleft()
                du = dist[u]
                if du >= half_girth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        cycle_len = du + dist[v] + 1
                        if cycle_len < required_girth:
                            return False
        return True

    def build_edges(m, voltages):
        """Build edge list for the lifted graph."""
        n = 6 * m
        edges = []
        for i in range(3):
            for j in range(3):
                v = voltages[i][j]
                for s in range(m):
                    u = i * m + s
                    w = 3 * m + j * m + (s + v) % m
                    edges.append((u, w))
        return n, edges

    # Strategy: try m from m_min upward, with multiple random voltage trials per m.
    # Use time budget to decide how many trials to run.
    random.seed(42)

    # For small m, try more trials; for larger m, fewer trials since each check is more expensive.
    for m in range(m_min, m_min + 300):
        if time.time() - start_time > time_limit - 30:  # Stop if too little time left
            break
        if 6 * m > 3000:  # Cap graph size
            break

        # Number of trials: more for smaller m, fewer for larger m
        trials = max(10, min(100, 5000 // m))

        for trial in range(trials):
            if time.time() - start_time > time_limit - 10:
                break

            # Generate random voltages
            voltages = [[random.randint(0, m - 1) for _ in range(3)] for _ in range(3)]

            if check_girth_fast(m, voltages, g):
                n, edges = build_edges(m, voltages)
                return n, edges

    # Fallback: use a known good construction or return a valid (possibly non-optimal) graph
    # Use a simple cycle-based construction as last resort
    # For 3-regular: use a "ladder" or "prism" like structure
    # Actually, let's try a few more m values with structured voltages

    # Structured voltage attempts for a few m values
    for m in [m_min, m_min + 5, m_min + 10, m_min + 20, m_min + 50, m_min + 100]:
        if time.time() - start_time > time_limit - 5:
            break
        if 6 * m > 3000:
            break

        # Try some structured patterns
        structured = [
            [[i * j % m for j in range(3)] for i in range(3)],
            [[(i + j) % m for j in range(3)] for i in range(3)],
            [[i * 2 * j % m for j in range(3)] for i in range(3)],
            [[(i * 3 + j) % m for j in range(3)] for i in range(3)],
        ]

        for voltages in structured:
            if time.time() - start_time > time_limit - 5:
                break
            if check_girth_fast(m, voltages, g):
                n, edges = build_edges(m, voltages)
                return n, edges

    # Last resort: return a valid 3-regular graph (may not meet girth, but should be valid structure)
    # Use a simple construction: 3-regular graph on 6 vertices (K_{3,3}) if g <= 4, else extend
    # For safety, use the L-U-W fallback from the seed
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
# EVOLVE-BLOCK-END
