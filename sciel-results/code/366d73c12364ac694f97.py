import time
import random
import math
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0

    # Fallback LUW Construction with Parameter Search
    def luw_construct(k, g, d_override=None):
        q = k
        if d_override is None:
            d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        else:
            d = d_override
            
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
        
        # Verify regularity and girth
        n = len(order)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Check degree
        for i in range(n):
            if len(adj[i]) != k:
                return None, None, False

        # Check girth
        if not check_girth(adj, n, g):
            return None, None, False
            
        return n, edges, True

    def check_girth(adj, n, g):
        max_depth = (g - 1) // 2
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                du = dist[u]
                if du >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        q.append(v)
                    elif v != s and dist[v] <= du:
                        cycle_len = du + dist[v] + 1
                        if cycle_len < g:
                            return False
        return True

    best_n = float('inf')
    best_edges = None

    # Search for optimal d
    # d must be at least 3
    # Max d such that n is reasonable
    max_d = 10 
    for d in range(3, max_d + 1):
        if time.time() > deadline - 20:
            break
        try:
            n, edges, valid = luw_construct(k, g, d)
            if valid and n < best_n:
                best_n = n
                best_edges = edges
        except Exception:
            pass

    if best_edges is None:
        # Fallback to original logic if search fails
        d_orig = max(3, g - 5 if (g - 5) % 2 else g - 4)
        n, edges, valid = luw_construct(k, g, d_orig)
        if valid:
            best_n = n
            best_edges = edges
        else:
            # Absolute fallback: Random graph? No, must be valid.
            # Return empty or raise?
            # The task requires a valid graph.
            # If LUW fails, we are in trouble.
            # But LUW is generally reliable.
            return 0, []

    return best_n, best_edges

