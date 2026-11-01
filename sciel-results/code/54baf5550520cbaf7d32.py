"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(k, g):
    import random
    import math
    import numpy as np
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
    
    # Strategy: try random voltage assignments for various m values,
    # check girth efficiently using non-backtracking walks, and return the smallest m that works.
    
    # Compute Moore bound
    if g % 2 == 0:
        moore_bound = 3 * (2 ** (g // 2 - 1) - 1)
    else:
        moore_bound = 3 * (2 ** ((g - 1) // 2 - 1) - 1) if (g - 1) // 2 >= 1 else 3
    
    m_min = max(1, math.ceil(moore_bound / 6))
    
    # Time limit: 120 seconds
    start_time = time.time()
    time_limit = 115  # Leave some margin
    
    def check_girth_fast(m, voltages, required_girth):
        """Check if the lifted graph has girth >= required_girth using non-backtracking BFS.
        voltages: 3x3 numpy array where voltages[i][j] is the voltage on edge from A_i to B_j.
        Returns True if girth >= required_girth, False otherwise.
        """
        n = 6 * m
        half_girth = required_girth // 2
        
        # For each vertex, do a non-backtracking BFS up to depth half_girth - 1
        # If we can reach any vertex at depth half_girth - 1 that has an edge back to a vertex
        # at depth < half_girth - 1 (not the parent), we have a short cycle.
        
        # Actually, simpler: do BFS from each vertex, tracking (vertex, prev_vertex) pairs.
        # If we encounter (v, u) where we've already seen (u, v), we have a cycle.
        # The cycle length is the sum of depths + 1.
        
        # More efficient: from each start vertex, do a tree search up to depth half_girth - 1
        # without backtracking. If the number of distinct vertices reached at depth d is less than
        # the number of non-backtracking walks of length d from the start, we have a collision
        # which implies a cycle.
        
        # For 3-regular bipartite graph, from any vertex, the number of non-backtracking walks
        # of length d is 3 * 2^(d-1) for d >= 1.
        
        # Let's use a different approach: for each starting vertex, do BFS but track the
        # previous vertex to avoid immediate backtracking. If we reach a vertex that was
        # already reached from a different path at the same or smaller depth, we have a cycle.
        
        # Actually, the cleanest way: for each start vertex s, do a BFS in the "line graph"
        # sense where states are (current, previous). Start with (s, None). Expand to
        # (v, u) for each neighbor v of u where v != prev. Track visited (v, u) pairs.
        # If we find that at depth d, two different paths lead to the same (v, u) state,
        # or if we can reach a state (v, u) where u was the start and d >= 2, we have a cycle.
        
        # Simpler still: just do BFS from each vertex, but when we see a visited neighbor
        # that isn't the parent, compute cycle length. This is O(n * e) per BFS but with
        # early stopping at half_girth depth.
        
        # For n up to ~1000 and half_girth = 8, this should be manageable.
        
        # Build adjacency
        adj = np.zeros((n, 3), dtype=np.int32)
        for i in range(3):
            for j in range(3):
                v = voltages[i][j]
                for s in range(m):
                    u = i * m + s
                    w = 3 * m + j * m + (s + v) % m
                    adj[u, j] = w
                    adj[w, i] = u
        
        # BFS from each vertex
        for start in range(n):
            # dist[v] = shortest distance from start to v
            # parent[v] = parent of v in BFS tree
            dist = np.full(n, -1, dtype=np.int32)
            parent = np.full(n, -1, dtype=np.int32)
            dist[start] = 0
            queue = deque([start])
            
            while queue:
                u = queue.popleft()
                du = dist[u]
                if du >= half_girth - 1:
                    continue
                for v in adj[u]:
                    v = int(v)
                    if dist[v] == -1:
                        dist[v] = du + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        # Cycle found: length = dist[u] + dist[v] + 1
                        cycle_len = du + dist[v] + 1
                        if cycle_len < required_girth:
                            return False
        
        return True
    
    random.seed(42)
    
    # Try m values from m_min upward
    # For each m, try random voltage assignments
    # Use a smarter search: start with larger m where success is more likely, then decrease
    
    # First, find any valid m (start from a reasonable upper bound and work down)
    # Or start from m_min and go up
    
    # Let's try from m_min upward, with more trials for smaller m
    best_result = None
    
    # Try structured voltages first for small m values
    # A good structured choice: voltages[i][j] = (a*i + b*j) % m for some a, b
    # or voltages[i][j] = (i * j * c + d) % m
    
    for m in range(m_min, min(m_min + 300, 200)):
        if 6 * m > 1200:  # Cap to keep check time reasonable
            break
        
        if time.time() - start_time > time_limit:
            break
        
        # Try structured voltage patterns first
        for a in range(1, min(m, 20)):
            for b in range(0, min(m, 20)):
                if time.time() - start_time > time_limit:
                    break
                voltages = np.zeros((3, 3), dtype=np.int64)
                for i in range(3):
                    for j in range(3):
                        voltages[i][j] = (a * i + b * j) % m
                
                if check_girth_fast(m, voltages, g):
                    n = 6 * m
                    edges = []
                    for i in range(3):
                        for j in range(3):
                            v = int(voltages[i][j])
                            for s in range(m):
                                u = i * m + s
                                w = 3 * m + j * m + (s + v) % m
                                edges.append((u, w))
                    return n, edges
        
        # Try random voltage assignments
        num_trials = max(10, 200 - m)  # More trials for smaller m
        for trial in range(num_trials):
            if time.time() - start_time > time_limit:
                break
            
            # Generate random voltages, ensuring some diversity
            voltages = np.random.randint(0, m, size=(3, 3)).astype(np.int64)
            
            if check_girth_fast(m, voltages, g):
                n = 6 * m
                edges = []
                for i in range(3):
                    for j in range(3):
                        v = int(voltages[i][j])
                        for s in range(m):
                            u = i * m + s
                            w = 3 * m + j * m + (s + v) % m
                            edges.append((u, w))
                return n, edges
    
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
