import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
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
        
        # Optimized Girth Check using BFS from each vertex
        # We only need to check if there is a cycle of length < g
        max_depth = g // 2
        
        for u in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            q = deque([u])
            
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
                        # The length of the cycle is dist[curr] + 1 + dist[v]
                        # But we must be careful: if v is in the same level or 
                        # previous level, the cycle length is dist[curr] + dist[v] + 1.
                        # However, in BFS, if v is visited, dist[v] <= dist[curr] + 1.
                        # If dist[v] == dist[curr] + 1, it's a child, but we handled parent.
                        # If dist[v] <= dist[curr], it's a cross edge or back edge.
                        # The cycle length is dist[curr] + dist[v] + 1.
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

        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        
        def get_rule(d_v):
            r = dict(rule)
            for c in range(5, d_v + 1):
                rc, c0 = (c - 5) % 4, c - (c - 5) % 4
                r[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][rc]
            return r

        r = get_rule(d_val)
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q_val):
                y = [None, first] + [0] * (d_val - 1)
                for c in range(2, d_val + 1):
                    a, b = r[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d_val)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    q = k
    
    # Try different d values
    # d must be large enough to ensure girth >= g
    # For LUW, girth is approx 2d+2 or similar.
    # We'll try a range.
    min_d = max(3, g // 2 - 1)
    
    for d_val in range(min_d, min_d + 10):
        if time.time() > deadline:
            break
        try:
            n, edges = build_luw_graph(q, d_val)
            # Optional: Skip if n is unreasonably large, but we want smallest.
            # If n < best_n, verify.
            if n < best_n:
                if verify_graph(n, edges, k, g):
                    best_n = n
                    best_edges = edges
                    # Since n increases with d, the first valid graph is the smallest.
                    # However, it's possible a smaller d produces an invalid graph (short cycles),
                    # and a larger d produces a valid one.
                    # We want the SMALLEST valid graph.
                    # So we should keep checking? No, n increases with d.
                    # So the first valid d gives the smallest n for this family.
                    # But what if a smaller d was invalid? We skipped it.
                    # So the first valid one is the best in this family.
                    break
        except Exception:
            pass

    if best_edges is None:
        # Fallback: If no valid graph found in the range, try a larger d
        # or return the smallest generated graph even if not verified (risky, but better than 0)
        # But the task requires a valid graph.
        # If we can't find one, we return 0, [] which scores 0.
        # Let's try one more large d just in case.
        try:
            d_val = min_d + 10
            n, edges = build_luw_graph(q, d_val)
            if verify_graph(n, edges, k, g):
                best_n = n
                best_edges = edges
        except Exception:
            pass

    if best_edges is None:
        return 0, []

    return best_n, best_edges
