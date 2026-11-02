import time
import math
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 150.0
    best_n = float('inf')
    best_edges = []
    
    # Target girth is g. We look for graphs with girth >= g.
    # LUW construction CD(d, q) has girth at least d + 5 for odd d.
    # We want girth >= g.
    # So d + 5 >= g => d >= g - 5.
    # If g is even, we can aim for girth g+1 (odd) or g.
    # LUW gives odd girth d+5.
    # So we set d such that d+5 >= g.
    # d = max(3, g - 5) if (g - 5) is odd?
    # If g=16, d+5 >= 16 => d >= 11.
    # If d=11, girth is at least 16.
    # The actual girth might be larger.
    
    d = max(3, g - 5)
    if d % 2 == 0:
        d += 1 # Ensure d is odd for the standard LUW girth guarantee?
        # Actually, the seed logic:
        # d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        # If g=16, g-5=11 (odd), so d=11.
        # If g=17, g-5=12 (even), so d=13.
        
    # Let's use the seed's d logic
    d_seed = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    # Try primes q
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    def build_luw(q, d):
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

    def check_girth(n, edges, min_girth):
        # BFS from every vertex to find shortest cycle
        # This is O(n * (n + m)) which is too slow for large n.
        # We can use non-backtracking walks.
        # Or just check if any cycle < min_girth exists.
        # For n < 1000, BFS from each vertex is feasible.
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            return False
        return True

    for q in primes:
        if time.time() > deadline - 10:
            break
            
        n, edges = build_luw(q, d_seed)
        
        # Quick check: if n is already larger than best, skip girth check?
        # No, we want the smallest n.
        # But girth check is expensive.
        # We can estimate girth.
        
        # For small n, run check_girth
        if n < 2000:
            if check_girth(n, edges, g):
                if n < best_n:
                    best_n = n
                    best_edges = edges
        else:
            # If n is large, assume it's valid if it's the first one?
            # No, we need to be sure.
            # But if n is large, it's unlikely to be the best.
            pass

    if best_n == float('inf'):
        # Fallback to seed result
        n, edges = build_luw(k, d_seed)
        return n, edges

    return best_n, best_edges
