import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 110s to be safe
    
    def verify_girth(adj, n, g):
        """Verify that the graph has girth at least g using BFS from each vertex."""
        depth_limit = (g - 1) // 2
        for u in range(n):
            visited = [-1] * n
            parent = [-1] * n
            visited[u] = 0
            queue = deque([(u, 0)])
            while queue:
                v, depth = queue.popleft()
                if depth >= depth_limit:
                    continue
                for w in adj[v]:
                    if w == parent[v]:
                        continue
                    if visited[w] == -1:
                        visited[w] = depth + 1
                        parent[w] = v
                        queue.append((w, depth + 1))
                    else:
                        # Found a cycle: path u->...->v->w->...->u
                        # Length = (depth + 1) + visited[w]
                        if (depth + 1) + visited[w] < g:
                            return False
        return True

    def luw_fallback():
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

    # For k=5, g=9: try to construct a graph with n around 110-200
    # Moore bound for k=5, g=9 (odd): n >= k + k*sum_{i=0}^{(g-3)/2} (k-1)^i = 5 + 5*(1+4+16) = 110
    
    def try_construct(n, max_attempts=50):
        """Try to construct a k-regular graph of girth >= g on n vertices."""
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Generate random k-regular graph using configuration model
            stubs = []
            for i in range(n):
                for _ in range(k):
                    stubs.append(i)
            
            random.shuffle(stubs)
            
            adj = [[] for _ in range(n)]
            valid = True
            for i in range(0, len(stubs), 2):
                u = stubs[i]
                v = stubs[i+1]
                if u == v:
                    valid = False
                    break
                if v in adj[u]:
                    valid = False
                    break
                adj[u].append(v)
                adj[v].append(u)
            
            if not valid:
                continue
            
            # Check girth
            if verify_girth(adj, n, g):
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges
        
        return None

    # Try increasing n from Moore bound upward
    moore_bound = k + k * sum((k-1)**i for i in range((g-3)//2 + 1))
    if g % 2 == 0:
        moore_bound = 2 + k * sum((k-1)**i for i in range(g//2 - 1))
    
    for n in range(moore_bound, min(moore_bound + 200, 300)):
        if time.time() > deadline:
            break
        
        result = try_construct(n, max_attempts=20)
        if result is not None:
            return result

    # Fallback to LUW
    return luw_fallback()
