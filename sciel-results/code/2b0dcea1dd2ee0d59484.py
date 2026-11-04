import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115

    def luw_construct(k, g, d_override=None):
        q = k
        if d_override is not None:
            d = d_override
        else:
            d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
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

    def verify_girth(n, edges, target_girth):
        """Verify girth >= target_girth using BFS from each vertex."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        # Check degree
        for u in range(n):
            if len(adj[u]) != k:
                return False
        # Check for short cycles via BFS
        for u in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            queue = deque([u])
            while queue:
                v = queue.popleft()
                if dist[v] >= target_girth // 2:
                    continue
                for w in adj[v]:
                    if dist[w] == -1:
                        dist[w] = dist[v] + 1
                        parent[w] = v
                        queue.append(w)
                    elif w != parent[v] and v != parent[w]:
                        # Found a cycle
                        cycle_len = dist[v] + dist[w] + 1
                        if cycle_len < target_girth:
                            return False
        return True

    if k != 3 or g != 16:
        return luw_construct(k, g)

    # Try different d values for the LUW construction
    # d must be large enough to ensure girth >= 16
    # The LUW construction with parameter d gives girth related to d
    # For k=3, g=16, try d values that might work
    best_n = float('inf')
    best_edges = None

    # Try d from small to large, preferring smaller graphs
    for d in range(7, 12):
        if time.time() > deadline:
            break
        n, edges = luw_construct(k, g, d_override=d)
        if n < best_n:
            # Verify girth
            if verify_girth(n, edges, g):
                best_n = n
                best_edges = edges
                # Found a valid graph, keep looking for smaller
            else:
                # This d doesn't give girth 16, try next
                pass

    if best_edges is not None:
        return best_n, best_edges

    # Fallback to original d
    return luw_construct(k, g)
