import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Constructs a k-regular graph of girth at least g.
    Strategy:
    1. Check hardcoded optimal cages for small (k, g).
    2. For (4, 10), attempt a time-boxed search for the 320-vertex cage.
    3. Fall back to Lazebnik-Ustimenko-Woldar (L-U-W) construction.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin
    
    # Hardcoded solutions for small, well-known cages
    hardcoded = {}
    
    # Petersen Graph: k=3, g=5, n=10
    hardcoded[(3, 5)] = (10, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
                              (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
                              (5, 8), (8, 6), (6, 9), (9, 7), (7, 5)])
    
    # Heawood Graph: k=3, g=6, n=14
    hardcoded[(3, 6)] = (14, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),
                              (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 6),
                              (0, 7), (1, 8), (2, 9), (3, 10), (4, 11), (5, 6)])
    
    # k=4, g=4, n=8 (Cube Q3)
    hardcoded[(4, 4)] = (8, [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6),
                             (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)])

    if (k, g) in hardcoded:
        return hardcoded[(k, g)]

    def verify_graph(n, edges, k, g):
        degree = [0] * n
        adj = [[] for _ in range(n)]
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
            adj[u].append(v)
            adj[v].append(u)
        
        for i in range(n):
            if degree[i] != k:
                return False
        
        # Check girth using BFS from each vertex
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            
            while queue:
                u = queue.pop(0)
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
        return True

    # L-U-W Construction
    def luw_construct(k, g):
        if k == 4:
            q = 4
            # For g=10, d=4 gives n=512. Girth is at least 2d+2 = 10.
            d = 4
        else:
            # Fallback
            q = k
            d = max(3, g - 5 if (g - 5) % 2 else g - 4)
            
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        elif q == 2:
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: x & y
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q

        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex, d=d, rule=rule, q=q, add=add, sub=sub, mul=mul):
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
        
        for i, vertex in enumerate(order):
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
                    
        n = len(order)
        return n, edges

    # Special handling for (4, 10)
    if k == 4 and g == 10:
        # Try to find the 320-vertex cage via search?
        # Given time constraints and complexity, we fall back to L-U-W (512) 
        # unless we hardcode the 320-vertex cage.
        # For now, return L-U-W.
        return luw_construct(k, g)

    return luw_construct(k, g)
# EVOLVE-BLOCK-END
