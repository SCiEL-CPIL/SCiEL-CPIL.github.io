import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115
    
    def luw_construction():
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

    def verify_girth(n, edges, g):
        """Verify using non-backtracking walks that no cycle is shorter than g."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check degree
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        # Non-backtracking walk verification
        # A non-backtracking walk is a sequence of vertices where consecutive vertices are adjacent
        # and no vertex immediately precedes its predecessor
        # We check that no non-backtracking walk of length < g returns to its start
        
        for start in range(n):
            # State: (current_vertex, previous_vertex)
            visited = set()
            queue = deque()
            for neighbor in adj[start]:
                queue.append((neighbor, start))
            
            depth = 1
            while queue and depth < g:
                new_queue = deque()
                for (v, prev) in queue:
                    if v == start:
                        return False  # Found a cycle of length < g
                    if (v, prev) in visited:
                        continue
                    visited.add((v, prev))
                    for w in adj[v]:
                        if w != prev:
                            new_queue.append((w, v))
                queue = new_queue
                depth += 1
        return True

    def try_local_search(k, g, deadline):
        """Try to find a smaller graph using local search."""
        # For k=4, g=10, the known minimum is 320
        # We'll try random 4-regular graphs with fewer vertices
        import random
        import math
        
        # Moore bound: 1 + k * sum_{i=0}^{g/2-1} (k-1)^i
        if g % 2 == 0:
            moore = 1 + k * sum((k-1)**i for i in range(g//2 - 1))
        else:
            moore = 1 + k * (sum((k-1)**i for i in range(g//2)) + (k-1)**(g//2))
        
        # Start from just above Moore bound
        for n in range(moore, 320):
            if time.time() > deadline - 10:
                break
            if n * k % 2 != 0:
                continue
            
            # Generate random k-regular graph
            for attempt in range(50):
                if time.time() > deadline - 10:
                    break
                try:
                    # Configuration model
                    half_edges = []
                    for i in range(n):
                        half_edges.extend([i] * k)
                    random.shuffle(half_edges)
                    
                    edges = set()
                    valid = True
                    for i in range(0, len(half_edges), 2):
                        u, v = half_edges[i], half_edges[i+1]
                        if u == v:
                            valid = False
                            break
                        if (min(u,v), max(u,v)) in edges:
                            valid = False
                            break
                        edges.add((min(u,v), max(u,v)))
                    
                    if not valid:
                        continue
                    
                    edge_list = list(edges)
                    if len(edge_list) != n * k // 2:
                        continue
                    
                    # Check girth
                    if verify_girth(n, edge_list, g):
                        return n, edge_list
                except Exception:
                    continue
        return None

    # First try local search for a smaller graph
    result = try_local_search(k, g, deadline)
    if result is not None:
        n, edges = result
        if verify_girth(n, edges, g):
            return n, edges
    
    # Fallback to LUW construction
    return luw_construction()
