import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 115.0 # 5s margin

    # Moore bound for k-regular graph with girth g (odd g=2r+1)
    # n >= 1 + k * sum_{i=0}^{r-1} (k-1)^i
    r = (g - 1) // 2
    moore_bound = 1 + k * sum((k-1)**i for i in range(r))

    def seed_construct(k, g, d_override=None):
        q = k
        # Default d calculation from current program
        if d_override is None:
            d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        else:
            d = d_override
            
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
        # Rule generation depends on d
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r_idx, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r_idx]

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
        
        # BFS to generate the component
        i = 0
        while i < len(order):
            if time.time() > deadline:
                break
            vertex = order[i]
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
            i += 1
        
        return len(order), edges

    def build_adj(edges, n):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def exact_girth_check(n, adj, g):
        """Check if girth is at least g."""
        half_g = g // 2
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                if dist[u] >= half_g:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        if v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return False
        return True

    # Strategy: Try different 'd' values for the L-U-W construction
    # The standard d is g-4 or g-5.
    # We try d, d-1, d-2, d-3 to see if a smaller graph with sufficient girth exists.
    
    best_n = float('inf')
    best_edges = None
    standard_d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    # Try d values in descending order of likely girth, but check size
    # Smaller d usually means smaller graph, but lower girth.
    # We need girth >= g.
    
    candidate_ds = [standard_d, standard_d - 1, standard_d - 2, standard_d - 3]
    candidate_ds = [d for d in candidate_ds if d >= 3]
    
    for d_val in candidate_ds:
        if time.time() > deadline - 5:
            break
            
        n, edges = seed_construct(k, g, d_val)
        
        if n < best_n:
            # Verify girth
            adj = build_adj(edges, n)
            if exact_girth_check(n, adj, g):
                best_n = n
                best_edges = edges
            # If n is already at Moore bound, we can't do better
            if best_n == moore_bound:
                break

    # Fallback to standard seed if no smaller valid graph found
    if best_edges is None:
        n, edges = seed_construct(k, g)
        best_n = n
        best_edges = edges

    return best_n, best_edges
