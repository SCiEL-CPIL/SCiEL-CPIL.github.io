import time
import random
import sys

def construct(k, g):
    # Helper to check girth efficiently
    def has_girth_at_least(n, edges, g):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check for simple graph
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            if (v, u) in edge_set:
                return False
            edge_set.add((u, v))
        
        # BFS from each vertex to find shortest cycle
        # Optimization: Only need to check up to distance g//2
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                if dist[u] >= g // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        # Found a back edge
                        if v != parent[u] and u != parent[v]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return False
        return True

    # Refactored LUW construction with explicit d
    def luw_construct(k, d):
        q = k
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
        # Rule generation for c in 2..d
        # The seed logic for rule was:
        # rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        # for c in range(5, d + 1):
        #    r, c0 = (c - 5) % 4, c - (c - 5) % 4
        #    rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        
        rule = {}
        if d >= 2: rule[2] = (1, 1)
        if d >= 3: rule[3] = (2, 1)
        if d >= 4: rule[4] = (1, 2)
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - r
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

    # Moore bound calculation
    D = (g - 2) // 2
    moore_bound = 1 + k * sum((k - 1) ** i for i in range(D))
    
    deadline = time.time() + 110
    
    # Strategy 1: Try LUW with smaller d to get smaller n
    # For k=4, g=10, d=4 gives n=512. d=5 gives n=2048.
    # We want to minimize n.
    # Try d from max(2, (g-2)//2 - 1) upwards?
    # Girth of LUW(d) is approx 2d+2.
    # We need 2d+2 >= g => d >= (g-2)/2.
    # For g=10, d >= 4.
    
    min_d = (g - 2) // 2
    if min_d < 2: min_d = 2
    
    # Try d = min_d first
    for d in range(min_d, min_d + 5):
        if time.time() > deadline:
            break
        n, edges = luw_construct(k, d)
        if has_girth_at_least(n, edges, g):
            return n, edges
            
    # Strategy 2: Random search for n close to Moore bound (if time left)
    # This is unlikely to succeed for g=10, k=4, but we try.
    if k == 4 and g == 10:
        # Try to find a 320 vertex graph?
        # Random 4-regular graph on 320 vertices.
        # Check girth.
        # If fail, do local search?
        # Given time constraints, we skip complex local search.
        pass

    # Fallback: LUW with larger d
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    return luw_construct(k, d)
