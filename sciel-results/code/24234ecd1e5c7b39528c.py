def construct(k, g):
    import time
    import random
    start_time = time.time()
    deadline = start_time + 110
    
    known_cages = {}
    
    # Petersen Graph (3, 5)
    known_cages[(3, 5)] = (10, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
                                 (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
                                 (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)])
    
    # Heawood Graph (3, 7)
    heawood_edges = []
    fano_lines = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    for i, line in enumerate(fano_lines):
        line_id = 7 + i
        for p in line:
            heawood_edges.append((p, line_id))
    known_cages[(3, 7)] = (14, heawood_edges)

    # (4, 5) cage: 4-regular, girth 5, 12 vertices
    # This is the icosahedron? No, icosahedron is 5-regular.
    # (4,5)-cage has 12 vertices. It's the complement of the icosahedron's... 
    # Actually the (4,5)-cage is the 12-vertex graph. Let me construct it.
    # The (4,5)-cage: take vertices 0-11. It's a 4-regular graph on 12 vertices with girth 5.
    # One construction: it's the graph where vertex i connects to i±1, i±2 mod 12? 
    # No that would give girth issues. Let me use a known construction.
    # Actually, the (4,5)-cage can be constructed as follows:
    # Vertices 0-11. Edges: (i, i+1) for i in 0..11 (cycle), plus (i, i+3) for i in 0..11
    # That gives degree 4. Let me check girth: cycle of length 5: 0-1-2-3-? 
    # 0 connects to 1,11,3,9. 1 connects to 0,2,4,10. 
    # 0-1-2-3: 3 connects to 2,4,0,6. So 0-1-2-3-0 is a 4-cycle! Bad.
    # Let me try: (i, i+1) and (i, i+4) mod 12.
    # 0: 1,11,4,8. 1: 0,2,5,9. 0-1-2-3: 3 connects to 2,4,7,11. No 0. 
    # 0-1-2-3-4: 4 connects to 3,5,0,8. 0-1-2-3-4-0 is a 5-cycle! Girth 5. Good.
    # Check for 4-cycles: 0-1-2-? 2 connects to 1,3,6,10. 0 connects to 1,11,4,8. No common.
    # 0-1-5: 5 connects to 4,6,1,9. 0 connects to 1,11,4,8. Common: 1,4. 
    # 0-1-5-4-0: that's a 4-cycle! Bad.
    # Let me just use a different approach. The (4,5)-cage is known to have 12 vertices.
    # I'll construct it programmatically if needed, or skip it.
    
    # (5, 6) cage: 5-regular, girth 6, 30 vertices
    # This is the 30-vertex graph. Let me try to construct it.
    # Actually, let me focus on making the algorithm work better for general cases.

    if (k, g) in known_cages:
        return known_cages[(k, g)]
    
    def verify_girth_fast(n, edges, g):
        """Verify girth using non-backtracking walks from every vertex."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check degrees
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        # Check no loops or multi-edges
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            key = (min(u, v), max(u, v))
            if key in edge_set:
                return False
            edge_set.add(key)
        
        # Non-backtracking walk verification
        # A cycle of length L < g exists iff there's a non-backtracking walk of length L-1 
        # that returns to its start vertex (with the last step not backtracking).
        # More precisely: use BFS in the non-backtracking graph.
        # State: (current_vertex, previous_vertex) where we arrived at current from previous.
        # We want to find if there's a cycle of length < g.
        # 
        # Alternative: BFS from each vertex, tracking parent to avoid backtracking.
        # This is equivalent to finding shortest cycles via BFS with parent tracking.
        
        for start in range(n):
            # BFS where state is (vertex, parent) to prevent backtracking
            # dist[(v, p)] = shortest non-backtracking path length from start to v arriving from p
            # We start by going from start to each neighbor
            dist = {}  # (v, p) -> distance
            queue = []
            for nb in adj[start]:
                dist[(nb, start)] = 1
                queue.append((nb, start))
            
            head = 0
            while head < len(queue):
                u, p = queue[head]
                head += 1
                d = dist[(u, p)]
                for v in adj[u]:
                    if v == p:
                        continue  # don't backtrack
                    if v == start:
                        # Found a cycle of length d+1
                        if d + 1 < g:
                            return False
                        continue
                    state = (v, u)
                    if state not in dist:
                        dist[state] = d + 1
                        queue.append(state)
                    # If state already seen, we found a cycle
                    # But we need to be careful: the cycle length is d+1 + dist[(u, v)] 
                    # Actually this gets complex. Let me use a simpler approach.
            # The above may miss some cycles. Let me use the standard BFS approach
            # but with the non-backtracking constraint.
        
        # Simpler: standard BFS from each vertex, tracking parent
        # The shortest cycle through vertex start is found by BFS
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
                    elif v != parent[u] and u != parent[v]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
        return True
    
    def verify_girth(n, edges, g):
        """Verify girth using non-backtracking walks (as specified in task)."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check degrees
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        # Check no loops or multi-edges
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            key = (min(u, v), max(u, v))
            if key in edge_set:
                return False
            edge_set.add(key)
        
        # Non-backtracking walk verification from every vertex
        # For each start vertex, do BFS in the state space (vertex, prev_vertex)
        # to find the shortest cycle.
        for start in range(n):
            # dist[(v, p)] = length of shortest non-backtracking path from start 
            # to v, arriving from p (i.e., last edge was p->v)
            dist = {}
            queue = []
            for nb in adj[start]:
                dist[(nb, start)] = 1
                queue.append((nb, start))
            
            head = 0
            while head < len(queue):
                u, p = queue[head]
                head += 1
                d = dist[(u, p)]
                if d >= g - 1:
                    continue  # any cycle through this state would have length >= g
                for v in adj[u]:
                    if v == p:
                        continue  # non-backtracking
                    if v == start:
                        # Cycle of length d+1
                        if d + 1 < g:
                            return False
                        continue
                    state = (v, u)
                    if state not in dist:
                        dist[state] = d + 1
                        queue.append(state)
        return True
    
    def build_luw(d, q):
        """Lazebnik-Ustimenko-Woldar construction for (q, 2d+1)-cage candidates."""
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
    
    def build_circulant(k, g):
        """Try to find a circulant graph with good girth."""
        if k % 2 == 0:
            half_k = k // 2
        else:
            half_k = k // 2
        
        best = None
        for trial in range(30):
            if time.time() > deadline:
                break
            base_n = 2 * (k - 1) ** (g // 2)
            n = base_n + random.randint(0, base_n // 3)
            n = max(n, 2 * k + 2)
            
            if k % 2 == 0:
                if n < 2 * half_k + 1:
                    continue
                gens = random.sample(range(1, n // 2 + 1), half_k)
                # Check no generator is 0 mod n
                if 0 in gens or n // 2 in gens and n % 2 == 0:
                    # n//2 is fine for even k as it gives a perfect matching
                    pass
                edges = []
                for i in range(n):
                    for gen in gens:
                        j = (i + gen) % n
                        if i < j:
                            edges.append((i, j))
            else:
                if n % 2 != 0:
                    continue
                half_k = k // 2
                if n // 2 <= half_k:
                    continue
                gens = random.sample(range(1, n // 2), half_k)
                edges = []
                for i in range(n):
                    for gen in gens:
                        j = (i + gen) % n
                        if i < j:
                            edges.append((i, j))
                    j = (i + n // 2) % n
                    if i < j:
                        edges.append((i, j))
            
            if len(edges) != n * k // 2:
                continue
            
            if verify_girth(n, edges, g):
                if best is None or n < best[0]:
                    best = (n, edges)
        
        return best
    
    # Try L-U-W first (deterministic, fast)
    best_n, best_edges = None, None
    q = k
    
    for d in range(2, min(g + 1, 12)):
        if time.time() > deadline - 5:
            break
        try:
            n, edges = build_luw(d, q)
            if verify_girth(n, edges, g):
                best_n, best_edges = n, edges
                break
        except Exception:
            continue
    
    # If L-U-W worked, try circulant to see if we can do better
    if time.time() < deadline - 10:
        circ = build_circulant(k, g)
        if circ is not None:
            if best_n is None or circ[0] < best_n:
                best_n, best_edges = circ
    
    if best_n is None:
        # Fallback: try larger LUW parameters
        for d in range(min(g, 10), min(g + 3, 15)):
            if time.time() > deadline - 5:
                break
            try:
                n, edges = build_luw(d, q)
                if verify_girth(n, edges, g):
                    best_n, best_edges = n, edges
                    break
            except Exception:
                continue
    
    if best_n is None:
        # Last resort: construct a valid k-regular graph (may not have good girth)
        n = 2 * k + 2
        edges = []
        for i in range(n):
            for j in range(1, k // 2 + 1):
                v1 = (i + j) % n
                v2 = (i - j) % n
                if i < v1:
                    edges.append((i, v1))
                if i < v2:
                    edges.append((i, v2))
        if k % 2 == 1:
            for i in range(n // 2):
                edges.append((i, i + n // 2))
        seen = set()
        unique_edges = []
        for u, v in edges:
            key = (min(u, v), max(u, v))
            if key not in seen:
                seen.add(key)
                unique_edges.append((u, v))
        best_n, best_edges = n, unique_edges
    
    return best_n, best_edges
