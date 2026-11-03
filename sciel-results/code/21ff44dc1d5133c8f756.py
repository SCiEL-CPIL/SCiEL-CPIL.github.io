import time
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    q = k
    # Try d values from small to large
    # Theoretical girth is d+5. We need girth >= g.
    # So d >= g - 5 is the safe lower bound.
    # But we can try d = g - 6, g - 7 etc. to see if actual girth is larger.
    # For k=3, g=16, safe d=11. Try 9, 10, 11.
    
    def build_luw(d):
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
        
        # Build adjacency list for girth check
        n = len(order)
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        return n, edges, adj

    def check_girth(n, adj, g):
        # Check if girth >= g
        # For each node, BFS to depth g//2 - 1
        max_depth = g // 2 - 1
        for s in range(n):
            if time.time() > deadline:
                return False # Timeout, assume invalid to be safe? Or return True? 
                             # If we assume invalid, we might skip a valid small graph.
                             # But if we run out of time, we should return the last known good one.
                             # This function should just return bool.
            
            visited = [-1] * n
            parent = [-1] * n
            queue = [(s, 0)]
            visited[s] = 0
            parent[s] = -2
            
            while queue:
                u, dist = queue.pop(0)
                if dist > max_depth:
                    continue
                for v in adj[u]:
                    if visited[v] == -1:
                        visited[v] = dist + 1
                        parent[v] = u
                        queue.append((v, dist + 1))
                    else:
                        if v != parent[u]:
                            # Found a back edge
                            # Cycle length estimate: dist[u] + dist[v] + 1
                            # If this is < g, definitely invalid
                            if dist + visited[v] + 1 < g:
                                return False
        return True

    best_result = None
    # Try d from small to large
    # Lower bound for d: g - 7 is a guess. g - 5 is safe.
    for d in range(max(3, g - 7), g - 3):
        if time.time() > deadline - 10:
            break
            
        try:
            n, edges, adj = build_luw(d)
        except Exception:
            continue
            
        # Quick check: if n is huge, skip? No, we want small n.
        # Check girth
        if check_girth(n, adj, g):
            best_result = (n, edges)
            break # Found a valid one with smallest d, so smallest n
    
    if best_result is None:
        # Fallback to safe d
        d_safe = max(3, g - 5 if (g - 5) % 2 else g - 4)
        try:
            n, edges, _ = build_luw(d_safe)
            best_result = (n, edges)
        except Exception:
            # Last resort: return empty or error?
            return 0, []
            
    return best_result
