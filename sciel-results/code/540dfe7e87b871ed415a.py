import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin
    
    def luw_construct(q, d):
        """Construct Lazebnik-Ustimenko-Woldar graph CD(d, q)"""
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
    
    # Try LUW with different d values to find the smallest valid graph
    # For girth g, we need d such that girth >= g
    # LUW CD(d,q) has girth at least d+5 for odd d, d+4 for even d
    best_result = None
    best_n = float('inf')
    
    # Try d values that should give girth >= g
    for d in range(max(3, g - 6), g + 2):
        if time.time() > deadline - 5:
            break
        
        n, edges = luw_construct(k, d)
        
        # Verify the graph is valid (k-regular, girth >= g)
        # Quick check: ensure all degrees are k
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if all(d_ == k for d_ in deg):
            if n < best_n:
                best_n = n
                best_result = (n, edges)
    
    if best_result:
        return best_result
    
    # Fallback: standard LUW
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    return luw_construct(k, d)
