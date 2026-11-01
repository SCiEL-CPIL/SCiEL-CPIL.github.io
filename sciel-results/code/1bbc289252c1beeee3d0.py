"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(k, g):
    import random
    import math
    from collections import deque
    import time

    # We construct a 3-regular graph using a voltage lift of K_{3,3} over Z_m.
    # The base graph is K_{3,3} with vertices A={0,1,2} and B={3,4,5}.
    # Edges in base: (i, j+3) for all i in 0..2, j in 0..2.
    # Each edge gets a voltage in Z_m. The lifted graph has 6*m vertices.
    # Vertex (side, base, shift) where side in {0,1}, base in 0..2, shift in 0..m-1.
    # Edge from (0, i, s) to (1, j, (s + v[i][j]) % m) where v[i][j] is the voltage.
    
    # For girth g, we need no cycles shorter than g.
    # The Moore bound for 3-regular graph with girth g gives lower bound on n.
    # For g=16, Moore bound is 512. The best known is 936.
    # With K_{3,3} lift, n = 6*m, so m = n/6.
    
    # Compute Moore bound
    if g % 2 == 0:
        moore_bound = 3 * (2 ** (g // 2 - 1) - 1)
    else:
        moore_bound = 3 * (2 ** ((g - 1) // 2 - 1) - 1) if (g - 1) // 2 >= 1 else 3
    
    m_min = max(1, math.ceil(moore_bound / 6))
    
    # Time budget
    start_time = time.time()
    time_limit = 100  # Leave 20s for safety
    
    def check_girth_fast(m, voltages, required_girth):
        """Fast girth check using non-backtracking BFS from each vertex.
        Returns True if girth >= required_girth, False otherwise.
        """
        n = 6 * m
        # Build adjacency list with edge indices to avoid backtracking
        # adj[u] = list of (v, edge_id)
        adj = [[] for _ in range(n)]
        
        for i in range(3):
            for j in range(3):
                v = voltages[i][j]
                for s in range(m):
                    u = i * m + s
                    w = 3 * m + j * m + (s + v) % m
                    adj[u].append(w)
                    adj[w].append(u)
        
        half_girth = required_girth // 2
        
        for start in range(n):
            # Non-backtracking BFS: state is (vertex, prev_vertex)
            # dist[state] = distance from start
            dist = {}
            queue = deque()
            
            # Initialize: from start, go to each neighbor
            for v in adj[start]:
                state = (v, start)
                dist[state] = 1
                queue.append(state)
            
            while queue:
                u, prev = queue.popleft()
                d = dist[(u, prev)]
                if d >= half_girth:
                    continue
                for v in adj[u]:
                    if v == prev:
                        continue
                    state = (v, u)
                    if state in dist:
                        # Found a cycle of length d + 1 + dist[state]
                        cycle_len = d + 1 + dist[state]
                        if cycle_len < required_girth:
                            return False
                    else:
                        dist[state] = d + 1
                        queue.append(state)
        
        return True
    
    def build_edges(m, voltages):
        """Build edge list from voltage assignment."""
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
    
    # Try random voltage assignments for increasing m
    random.seed(42)
    
    # For each m, try several random voltage assignments
    # Start with structured assignments, then random
    
    for m in range(m_min, m_min + 300):
        if time.time() - start_time > time_limit:
            break
        if 6 * m > 3000:  # Cap to avoid too large graphs
            break
        
        # Try a few structured assignments first
        # Diagonal voltages: v[i][j] = a*i + b*j + c (mod m)
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    voltages = [[(a * i + b * j + c) % m for j in range(3)] for i in range(3)]
                    if check_girth_fast(m, voltages, g):
                        return build_edges(m, voltages)
        
        # Try random voltage assignments
        num_trials = 30
        for trial in range(num_trials):
            if time.time() - start_time > time_limit:
                break
            # Generate random voltages
            voltages = [[random.randint(0, m - 1) for _ in range(3)] for _ in range(3)]
            
            if check_girth_fast(m, voltages, g):
                return build_edges(m, voltages)
    
    # Fallback: use the Lazebnik-Ustimenko-Woldar construction
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
