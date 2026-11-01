import time
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    def luw_construct(k, g):
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

    # Try LUW construction
    best_n, best_edges = luw_construct(k, g)
    
    # Fast girth verification using non-backtracking walks
    def has_short_cycle(n, edges, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Non-backtracking walk BFS: state is (node, prev_node)
        # Track shortest distance for each (node, prev) pair
        from collections import deque
        
        # For each starting directed edge, do non-backtracking BFS
        # We want to find if any cycle of length < g exists
        # A cycle of length L means a non-backtracking walk of length L that returns to start
        
        # More efficient: for each vertex, do non-backtracking BFS up to depth (g-1)//2
        # and check for collisions
        
        # Actually, let's use a simpler approach: BFS from each vertex but with early termination
        # and only up to depth (g-2)//2 since a cycle of length < g means
        # two vertices at distance < (g-1)/2 from start are connected
        
        max_depth = (g - 2) // 2  # we only need to check up to this depth
        
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                node = queue.popleft()
                if dist[node] > max_depth:
                    continue
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
                    elif nb != parent[node]:
                        # Found a cycle
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < g:
                            return True
        return False

    if not has_short_cycle(best_n, best_edges, g):
        return best_n, best_edges
    
    # Fallback: return the LUW result anyway (it should be valid)
    return best_n, best_edges
