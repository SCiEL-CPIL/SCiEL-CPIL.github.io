import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_n = float('inf')
    best_edges = None

    def verify_graph(n, edges, k, g):
        """
        Verify that the graph is:
        1. Simple (no loops, no multi-edges)
        2. k-regular
        3. Has girth >= g
        """
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
        
        # Girth Check using BFS from each vertex with early termination
        # We only need to check if there is a cycle of length < g
        max_depth = g // 2
        
        for u in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            q = deque([u])
            found_short_cycle = False
            
            while q:
                curr = q.popleft()
                if dist[curr] >= max_depth:
                    continue
                
                for v in adj[curr]:
                    if dist[v] == -1:
                        dist[v] = dist[curr] + 1
                        parent[v] = curr
                        q.append(v)
                    else:
                        # Ignore tree edges (parent-child)
                        if v == parent[curr]:
                            continue
                        # If v is already visited and not parent, 
                        # it forms a cycle.
                        cycle_len = dist[curr] + dist[v] + 1
                        if cycle_len < g:
                            return False
                            
        return True

    def build_luw_graph(q_val, d_val):
        # LUW Construction
        if q_val == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q_val
            sub = lambda x, y: (x - y) % q_val
            mul = lambda x, y: x * y % q_val

        # Rules for LUW construction
        # For c=2: (1,1), c=3: (2,1), c=4: (1,2)
        # For c>=5, use a pattern that ensures girth
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        
        def get_rule(d_v):
            r = dict(rule)
            # For c >= 5, use rules that avoid short cycles
            # Pattern: cycle through (1,c-2), (c-1,1), (c,1), (1,c+1)
            for c in range(5, d_v + 1):
                rc = (c - 5) % 4
                c0 = c - (c - 5) % 4
                if rc == 0:
                    r[c] = (1, c0 - 2)
                elif rc == 1:
                    r[c] = (c0 - 1, 1)
                elif rc == 2:
                    r[c] = (c0, 1)
                else:
                    r[c] = (1, c0 + 1)
            return r

        r = get_rule(d_val)
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q_val):
                y = [None, first] + [0] * (d_val - 1)
                for c in range(2, d_val + 1):
                    a, b = r[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d_val)
        index = {start: 0}
        order = [start]
        edges = []
        
        # BFS to build the graph
        for i, vertex in enumerate(order):
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        return len(order), edges

    q = k
    
    # For k=4, g=10: Moore bound is 243, known cage is 320
    # LUW graphs for q=4 have n = q * (q^d - 1)/(q-1) = 4 * (4^d - 1)/3
    # d=4: n=4*(64-1)/3=84, d=5: n=4*(1024-1)/3=1364
    # We need girth >= 10, so d should be at least 4 or 5
    
    # Try d values starting from a reasonable minimum
    min_d = max(3, (g - 2) // 2)  # For g=10, this gives 4
    
    # Search range: try d from min_d to min_d+15
    for d_val in range(min_d, min_d + 16):
        if time.time() > deadline - 5:  # Leave 5s for verification
            break
        try:
            n, edges = build_luw_graph(q, d_val)
            
            # Quick check: if n is already larger than best, skip
            if n >= best_n:
                continue
            
            if verify_graph(n, edges, k, g):
                best_n = n
                best_edges = edges
                # Since n increases with d for LUW, the first valid d gives smallest n
                break
        except Exception:
            pass

    if best_edges is None:
        # Fallback: try larger d values
        for d_val in range(min_d + 16, min_d + 30):
            if time.time() > deadline - 5:
                break
            try:
                n, edges = build_luw_graph(q, d_val)
                if n < best_n and verify_graph(n, edges, k, g):
                    best_n = n
                    best_edges = edges
                    break
            except Exception:
                pass

    if best_edges is None:
        return 0, []

    return best_n, best_edges
