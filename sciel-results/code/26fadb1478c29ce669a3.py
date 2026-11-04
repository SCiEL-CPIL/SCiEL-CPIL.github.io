import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    
    def check_girth(n, edges, target_girth):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        min_cycle = float('inf')
        
        for i in range(n):
            dist = [-1] * n
            parent = [-1] * n
            q = deque()
            q.append(i)
            dist[i] = 0
            parent[i] = -2
            
            while q:
                u = q.popleft()
                du = dist[u]
                if du * 2 + 1 > min_cycle: # Pruning
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u]:
                        cycle_len = du + dist[v] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                            if min_cycle < target_girth:
                                return min_cycle
            if min_cycle < target_girth:
                return min_cycle
        return min_cycle

    def generate_luw(k, g, q, d):
        if q == 2:
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: x & y
        elif q == 3:
            add = lambda x, y: (x + y) % 3
            sub = lambda x, y: (x - y) % 3
            mul = lambda x, y: x * y % 3
        elif q == 4:
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
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
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

    if k == 3 and g == 16:
        # Try q=2
        n, edges = generate_luw(3, 16, 2, 11)
        
        # Check girth
        # If time is running out, skip check? No, we need validity.
        # But N=4096 is small.
        girth_found = check_girth(n, edges, 16)
        
        if girth_found >= 16:
            return n, edges
            
        # Fallback to q=3
        n, edges = generate_luw(3, 16, 3, 11)
        return n, edges

    # General case
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    n, edges = generate_luw(k, g, q, d)
    return n, edges
