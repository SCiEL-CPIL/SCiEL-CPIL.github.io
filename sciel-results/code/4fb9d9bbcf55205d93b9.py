import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    def verify_graph(n, edges, k, g, time_limit=10.0):
        """
        Verify that the graph is:
        1. Simple (no loops, no multi-edges)
        2. k-regular
        3. Has girth >= g (using BFS-based girth check with time limit)
        """
        if time.time() > start_time + time_limit:
            return False

        if n == 0:
            return False
        
        # Check degree and build adjacency list
        adj = [[] for _ in range(n)]
        seen_edges = set()
        
        for u, v in edges:
            if u == v:
                return False
            if u < 0 or u >= n or v < 0 or v >= n:
                return False
            edge_key = (min(u, v), max(u, v))
            if edge_key in seen_edges:
                return False
            seen_edges.add(edge_key)
            adj[u].append(v)
            adj[v].append(u)
        
        # Check regularity
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        # Girth check: find shortest cycle using BFS from each vertex
        # A cycle of length L exists iff there's a non-tree edge (u,v) where
        # dist[u] + dist[v] + 1 = L in a BFS tree from some root.
        # We only need to check if any cycle has length < g.
        max_depth = g // 2  # If shortest cycle < g, it will be found within depth g//2
        
        for u in range(n):
            if time.time() > start_time + time_limit:
                return False
                
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            q = deque([u])
            
            while q:
                curr = q.popleft()
                if dist[curr] >= max_depth:
                    continue
                
                for v in adj[curr]:
                    if v == parent[curr]:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[curr] + 1
                        parent[v] = curr
                        q.append(v)
                    else:
                        # Found a back/cross edge
                        cycle_len = dist[curr] + dist[v] + 1
                        if cycle_len < g:
                            return False
                            
        return True

    def build_luw_graph(q_val, d_val):
        """
        LUW construction for q-regular graphs.
        For q=4, uses GF(4) arithmetic.
        """
        if q_val == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        elif q_val == 3:
            add = lambda x, y: (x + y) % 3
            sub = lambda x, y: (x - y) % 3
            mul = lambda x, y: (x * y) % 3
        elif q_val == 5:
            add = lambda x, y: (x + y) % 5
            sub = lambda x, y: (x - y) % 5
            mul = lambda x, y: (x * y) % 5
        elif q_val == 7:
            add = lambda x, y: (x + y) % 7
            sub = lambda x, y: (x - y) % 7
            mul = lambda x, y: (x * y) % 7
        else:
            # Fallback: assume prime
            add = lambda x, y: (x + y) % q_val
            sub = lambda x, y: (x - y) % q_val
            mul = lambda x, y: (x * y) % q_val

        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        
        def get_rule(d_v):
            r = dict(rule)
            for c in range(5, d_v + 1):
                rc = (c - 5) % 4
                c0 = c - rc
                if rc == 0:
                    r[c] = (1, c0 - 2)
                elif rc == 1:
                    r[c] = (c0 - 1, 1)
                elif rc == 2:
                    r[c] = (c0, 1)
                elif rc == 3:
                    r[c] = (1, c0 + 1)
            return r

        r = get_rule(d_val)
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q_val):
                y = [0] * (d_val + 1)
                y[1] = first
                for c in range(2, d_val + 1):
                    a, b = r[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield (1 - side, tuple(y))

        start = (0, tuple([0] * (d_val + 1)))
        index = {start: 0}
        order = [start]
        edges = []
        
        for i, vertex in enumerate(order):
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        return len(order), edges

    def build_fallback(k_val, g_val):
        """
        Fallback construction for non-prime-power k or when LUW fails.
        Uses a simple circulant-like construction.
        """
        # For small cases, try a known graph
        # For k=4, g=10, we can try to build a 4-regular graph on ~320 vertices
        # using a structured approach
        
        # Simple approach: use a toroidal grid with perturbations
        # This is a rough fallback; the LUW should handle the main case
        
        n = k * (k - 1) ** ((g_val - 2) // 2)  # Moore bound approximation
        n = max(n, 10)
        
        # Build a k-regular graph using a simple construction
        # For now, return a placeholder that will fail verification
        # but is structurally valid
        edges = []
        for i in range(n):
            for j in range(1, k // 2 + 1):
                edges.append((i, (i + j) % n))
                edges.append((i, (i - j) % n))
        
        return n, edges

    best_n = float('inf')
    best_edges = None

    # Primary strategy: LUW construction
    # For k=4, g=10: use d=5 (girth 12) as the smallest valid LUW
    if k in [3, 4, 5, 7]:
        # Try d values starting from the minimum that gives girth >= g
        # LUW girth is approximately 2*d + 2, so we need 2*d + 2 >= g
        # => d >= (g - 2) / 2
        min_d = max(3, (g - 2) // 2)
        
        for d_val in range(min_d, min_d + 5):
            if time.time() > deadline:
                break
            try:
                n, edges = build_luw_graph(k, d_val)
                if n < best_n:
                    # Verify the graph
                    if verify_graph(n, edges, k, g, time_limit=15.0):
                        best_n = n
                        best_edges = edges
                        break  # Found a valid graph, stop
            except Exception:
                pass
    
    # If LUW didn't work or k is not a prime power, try fallback
    if best_edges is None:
        if time.time() < deadline:
            try:
                n, edges = build_fallback(k, g)
                if verify_graph(n, edges, k, g, time_limit=15.0):
                    best_n = n
                    best_edges = edges
            except Exception:
                pass

    if best_edges is not None:
        return best_n, best_edges

    # Last resort: return an empty graph (will score 0 but won't crash)
    return 0, []
