import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0

    # Moore bound for k-regular graph of girth g
    # For k=3, g=16: M = 2 * (2^8 - 1) = 510. So n >= 512 (even).
    if k == 3:
        if g % 2 == 0:
            moore_bound = 2 * sum((k-1)**i for i in range(g//2))
        else:
            moore_bound = (k-1) * sum((k-1)**i for i in range((g-1)//2))
        # Actually, for k=3, g=16:
        # M = 2 * (1 + 2 + 4 + ... + 2^7) = 2 * (2^8 - 1) = 510.
        # n must be even, so n >= 512.
    else:
        moore_bound = 1

    # Seed algebraic construction parameters
    q = k
    # The seed code used d = g-4 or g-5.
    # We will search for the smallest d such that 2*q^d >= moore_bound
    # and the resulting graph has girth >= g.
    
    def build_graph(d):
        """Build the graph using the seed's algebraic rules for a given depth d."""
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
        
        # Define rules for coefficients
        # The seed code had specific rules for c in 2..d
        # We reuse the seed's rule generation logic
        rule = {}
        if d >= 2: rule[2] = (1, 1)
        if d >= 3: rule[3] = (2, 1)
        if d >= 4: rule[4] = (1, 2)
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

    def check_girth(edges, n, g):
        """Check if graph has girth >= g."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        half_g = g // 2
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= half_g:
                    break
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
                    elif v != start and dist[u] + dist[v] + 1 < g:
                        # Found a cycle shorter than g
                        return False
        return True

    # Search for the smallest d
    best_n = float('inf')
    best_edges = None
    
    # Minimum d such that 2*q^d >= moore_bound
    min_d = 1
    while 2 * (q ** min_d) < moore_bound:
        min_d += 1
        
    # Try d from min_d upwards
    # Limit d to a reasonable range to avoid huge graphs if girth check fails
    max_d = min_d + 5 
    
    for d in range(min_d, max_d + 1):
        if time.time() > deadline - 10:
            break
            
        n, edges = build_graph(d)
        
        # Quick check: if n is already larger than our best, skip (but we want smallest n)
        # We are iterating d increasing, so n is increasing.
        # The first valid graph is the smallest in this family.
        
        if check_girth(edges, n, g):
            best_n = n
            best_edges = edges
            break # Found the smallest valid one in this family
            
    # If no valid graph found in the small d range, fall back to the seed's original large d
    if best_edges is None:
        d_fallback = max(3, g - 5 if (g - 5) % 2 else g - 4)
        best_n, best_edges = build_graph(d_fallback)
        
    return best_n, best_edges
