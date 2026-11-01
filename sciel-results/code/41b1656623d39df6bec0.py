"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(k, g):
    import random
    import math
    import time
    from collections import deque

    # Construct a 3-regular graph via voltage lift of K_{3,3} over Z_m.
    # n = 6*m vertices. Use non-backtracking BFS for fast girth checking.

    if g % 2 == 0:
        moore_bound = 3 * (2 ** (g // 2 - 1) - 1)
    else:
        moore_bound = 3 * (2 ** ((g - 1) // 2 - 1) - 1) if (g - 1) // 2 >= 1 else 3

    m_min = max(1, math.ceil(moore_bound / 6))
    time_limit = 115.0
    start_time = time.time()

    def build_adj(m, voltages):
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
        return n, adj

    def check_girth(m, voltages, required_girth):
        """Check girth >= required_girth using per-vertex BFS with depth limit."""
        n, adj = build_adj(m, voltages)
        max_depth = required_girth // 2
        
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            
            while queue:
                u = queue.popleft()
                du = dist[u]
                if du >= max_depth:
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

    random.seed(42)
    
    # Try m values from m_min upward
    # For each m, try multiple random voltage assignments
    # Use time limit to stop
    
    # Precompute some structured voltage patterns for better initial coverage
    def gen_random_voltages(m):
        return [[random.randint(0, m - 1) for _ in range(3)] for _ in range(3)]

    best_result = None
    
    for m in range(m_min, min(m_min + 300, 400)):
        if 6 * m > 2000:
            break
        if time.time() - start_time > time_limit:
            break
        
        # Number of trials decreases as we go higher in m
        if m < m_min + 30:
            num_trials = 30
        elif m < m_min + 80:
            num_trials = 15
        else:
            num_trials = 8
        
        for trial in range(num_trials):
            if time.time() - start_time > time_limit:
                break
            
            voltages = gen_random_voltages(m)
            
            # Quick check: avoid trivially bad patterns
            # Check that no voltage is 0 (which could create short cycles)
            # Actually, let's just run the check
            
            if check_girth(m, voltages, g):
                # Build the graph
                n, adj = build_adj(m, voltages)
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges
    
    # Fallback: Lazebnik-Ustimenko-Woldar construction
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
