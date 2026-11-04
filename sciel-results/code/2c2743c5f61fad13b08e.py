import time
import random
import math

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    def check_girth(n, edges, min_girth):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # BFS from each vertex to find shortest cycle
        # We only need to check up to depth floor((min_girth-1)/2)
        max_depth = (min_girth - 1) // 2
        
        for start in range(n):
            # dist[v] = distance from start
            # parent[v] = parent in BFS tree
            dist = [-1] * n
            parent = [-1] * n
            queue = [start]
            dist[start] = 0
            parent[start] = -2 # sentinel
            
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                if dist[u] >= max_depth:
                    continue
                
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        # Found a cycle
                        # Length is dist[u] + dist[v] + 1
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            return False
        return True

    def luw_construction(k, d):
        q = k
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
        # Generate rules
        rule = {}
        rule[2] = (1, 1)
        rule[3] = (2, 1)
        rule[4] = (1, 2)
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

    # Determine minimum d to try
    # Moore bound radius is floor((g-1)/2)
    min_d = max(3, (g - 1) // 2)
    
    best_result = None
    best_n = float('inf')
    
    # Try d from min_d upwards
    # We cap d to avoid infinite loop, e.g., d <= 10
    for d in range(min_d, min_d + 5):
        if time.time() > deadline:
            break
            
        try:
            n, edges = luw_construction(k, d)
        except Exception:
            continue
            
        if n < best_n:
            # Check girth
            # Optimization: If n is very large, check_girth might be slow.
            # But for k=5, d=4, n=1250. BFS is fast.
            if check_girth(n, edges, g):
                best_n = n
                best_result = (n, edges)
                # If we found a graph close to Moore bound, we might stop?
                # But smaller d is better. We are iterating d upwards.
                # So the first valid one is the smallest n from this family.
                break
    if best_result:
        return best_result

    # Fallback: Use the seed's logic (which likely used d=5 for g=9)
    # The seed's d calculation:
    d_seed = max(3, g - 5 if (g - 5) % 2 else g - 4)
    n, edges = luw_construction(k, d_seed)
    return n, edges
