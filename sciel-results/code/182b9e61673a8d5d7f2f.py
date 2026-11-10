import time
import random
import math
import sys

def _luw_construct(k, g, q=None, d=None):
    """LUW construction with optional q and d overrides."""
    if q is None:
        q = k
    if d is None:
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

def _check_girth_fast(adj, n, min_girth):
    """Check if graph has girth >= min_girth using BFS from each vertex. Early exit."""
    max_depth = min_girth // 2 - 1
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = [start]
        qi = 0
        while qi < len(queue):
            node = queue[qi]
            qi += 1
            d = dist[node]
            if d >= max_depth:
                continue
            for nb in adj[node]:
                if dist[nb] == -1:
                    dist[nb] = d + 1
                    queue.append(nb)
                elif dist[nb] <= d:
                    cycle_len = d + dist[nb] + 1
                    if cycle_len < min_girth:
                        return False
    return True

def construct(k, g):
    deadline = time.time() + 115
    
    # For k=4, g=10, try multiple LUW parameterizations
    if k == 4 and g == 10:
        candidates = []
        # Original: q=4, d=5
        n1, e1 = _luw_construct(4, 10, q=4, d=5)
        candidates.append((n1, e1))
        
        # q=2, d=8: n=2*2^8=512
        n2, e2 = _luw_construct(4, 10, q=2, d=8)
        candidates.append((n2, e2))
        
        # q=2, d=9: n=2*2^9=1024
        n3, e3 = _luw_construct(4, 10, q=2, d=9)
        candidates.append((n3, e3))
        
        # q=3, d=6: n=2*3^6=1458
        n4, e4 = _luw_construct(4, 10, q=3, d=6)
        candidates.append((n4, e4))
        
        # Pick the smallest
        candidates.sort(key=lambda x: x[0])
        best_n, best_edges = candidates[0]
        
        # Verify girth of the best candidate
        adj = [set() for _ in range(best_n)]
        for u, v in best_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        if _check_girth_fast(adj, best_n, g):
            return best_n, best_edges
        
        # If girth check fails, fall back to the original
        return n1, e1
    
    # Fallback for other k, g
    return _luw_construct(k, g)
