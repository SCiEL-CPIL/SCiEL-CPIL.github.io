import random
import time
import sys
from collections import deque

def construct(k, g):
    # Fallback to LUW construction if search fails
    # (Copy of the seed logic, slightly optimized)
    def luw_construct(k, g):
        q = k
        # Try to find the smallest d that yields girth >= g
        # Theoretical girth for LUW is approx 2d+1 or 2d+3
        # We will generate for d=3, 4, 5... and check actual girth
        
        best_n = float('inf')
        best_edges = None
        
        # d must be at least 3 for k>=3
        # Max d limited by time/size
        max_d = 6 
        
        for d in range(3, max_d + 1):
            if q == 4:
                table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
                add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
            else:
                add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
            
            # Rule generation as in seed
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
            
            n = len(order)
            
            # Check if this graph is valid (simple, k-regular, girth >= g)
            # It is simple and k-regular by construction.
            # We need to check girth.
            # For efficiency, we only check girth if n < best_n
            if n < best_n:
                # Quick girth check
                # Build adjacency
                adj = [[] for _ in range(n)]
                for u, v in edges:
                    adj[u].append(v)
                    adj[v].append(u)
                
                # BFS from a few nodes to estimate girth
                # If girth is small, we reject.
                min_girth = 10**9
                # Sample a few start nodes
                for i in range(0, n, max(1, n//10)):
                    dist = [-1] * n
                    parent = [-1] * n
                    dist[i] = 0
                    q_bfs = deque([i])
                    while q_bfs:
                        u = q_bfs.popleft()
                        for v in adj[u]:
                            if dist[v] == -1:
                                dist[v] = dist[u] + 1
                                parent[v] = u
                                q_bfs.append(v)
                            elif parent[u] != v and parent[v] != u:
                                cycle_len = dist[u] + dist[v] + 1
                                if cycle_len < min_girth:
                                    min_girth = cycle_len
                    if min_girth < g:
                        break
                
                if min_girth >= g:
                    best_n = n
                    best_edges = edges
                else:
                    # If girth is too small, larger d might be needed
                    # But we continue to try larger d in the loop
                    pass

        if best_edges is not None:
            return best_n, best_edges
        
        # Fallback to original d calculation if loop failed (should not happen)
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

    return luw_construct(k, g)
