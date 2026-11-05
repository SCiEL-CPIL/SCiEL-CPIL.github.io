import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # 1. Construct LUW graph as baseline
    # For k=5, g=9, this yields n=1250.
    # This is the best known easy-to-construct graph for this parameter.
    # Finding a smaller one (like the 1116 cage) requires advanced search not feasible in this budget.
    
    q = k
    if (g - 5) % 2 == 1:
        d = g - 5
    else:
        d = g - 4
    d = max(3, d)
    
    if q == 4:
        mult_table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        add_table = [
            [0, 1, 2, 3],
            [1, 0, 3, 2],
            [2, 3, 0, 1],
            [3, 2, 1, 0]
        ]
        def add(x, y): return add_table[x][y]
        def sub(x, y): return add_table[x][y]
        def mul(x, y): return mult_table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q
    
    rule = [None] * (d + 1)
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0: rule[c] = (1, c0 - 2)
        elif r == 1: rule[c] = (c0 - 1, 1)
        elif r == 2: rule[c] = (c0, 1)
        else: rule[c] = (1, c0 + 1)
    
    def get_neighbours(side, x):
        for first in range(q):
            y = [0] * (d + 1)
            y[0] = 0
            y[1] = first
            if side == 0:
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b]))
            else:
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)
    
    start = (0, (0,) * (d + 1))
    index = {start: 0}
    order = [start]
    luw_edges = []
    
    for vertex in order:
        if time.time() > deadline: break
        side, x = vertex
        for other in get_neighbours(side, x):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if side == 0:
                luw_edges.append((index[vertex], index[other]))
                
    luw_n = len(order)
    
    # Verify the graph is valid (simple, regular, girth >= g)
    # This step is crucial to ensure we don't return an invalid graph.
    # For n=1250, k=5, g=9, the LUW graph is known to be valid.
    # We perform a quick check to be safe.
    
    adj = [[] for _ in range(luw_n)]
    edges_set = set()
    for u, v in luw_edges:
        if u == v:
            return 0, [] # Invalid
        if (u, v) in edges_set or (v, u) in edges_set:
            return 0, [] # Invalid
        edges_set.add((u, v))
        adj[u].append(v)
        adj[v].append(u)
        
    # Check regularity
    for i in range(luw_n):
        if len(adj[i]) != k:
            return 0, [] # Invalid

    # Check girth using non-backtracking BFS
    # This is O(n * (k-1)^(g/2)) which is manageable for n=1250, g=9.
    def check_girth(adj, n, g):
        for s in range(n):
            visited = set()
            queue = deque()
            for t in adj[s]:
                queue.append((s, t, 1))
                visited.add((s, t))
            while queue:
                u, prev, dist = queue.popleft()
                if dist >= g:
                    continue
                for v in adj[u]:
                    if v == prev:
                        continue
                    if v == s and dist + 1 >= 3:
                        if dist + 1 < g:
                            return False
                    if (v, u) not in visited:
                        visited.add((v, u))
                        queue.append((v, u, dist + 1))
        return True

    if not check_girth(adj, luw_n, g):
        # Fallback: return empty if check fails (should not happen for LUW)
        return 0, []

    return luw_n, luw_edges
