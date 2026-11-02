from collections import deque

def construct(k, g):
    q = k
    # Target girth g.
    # L-U-W construction CD(d, q) has n = 2 * q^d.
    # Girth is typically 2d+2 or similar.
    # For g=9, d=4 gives girth >= 9. n = 2 * 5^4 = 1250.
    
    def build_luw(d):
        # Field operations for GF(q)
        # q is prime (5)
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q

        # L-U-W Rule
        rule = {}
        for c in range(2, d + 1):
            if c == 2:
                rule[c] = (1, 1)
            elif c == 3:
                rule[c] = (2, 1)
            elif c == 4:
                rule[c] = (1, 2)
            else:
                r, c0 = (c - 5) % 4, c - (c - 5) % 4
                rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def get_neighbours(vertex):
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
        index = {start: 0}
        order = [start]
        edges = []
        
        for vertex in order:
            for other in get_neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
                    
        n = len(order)
        
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Verify Girth
        girth = float('inf')
        for start_v in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start_v] = 0
            queue = deque([start_v])
            found_short_cycle = False
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < girth:
                            girth = cycle_len
                        if girth < g:
                            found_short_cycle = True
                            break
                if found_short_cycle:
                    break
            if found_short_cycle:
                break
                
        return n, edges, girth

    # Try d=4
    n, edges, girth = build_luw(4)
    if girth >= g:
        return n, edges
        
    # Fallback to d=5 if d=4 fails (unlikely)
    n, edges, girth = build_luw(5)
    if girth >= g:
        return n, edges
        
    raise ValueError("Construction failed")
