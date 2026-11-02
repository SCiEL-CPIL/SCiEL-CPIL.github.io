import time
import itertools
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Target: 4-regular, girth >= 10
    # Search for circulant graphs C_n(S)
    
    # Moore bound for (4,10) is 243.
    # n must be even.
    # Try n from 244 to 8000
    best_n = 8192 # Fallback to LUW size
    best_edges = None
    
    # We can also try to construct the LUW graph as a fallback
    # But let's try circulant first.
    
    def check_girth_fast(n, steps, g):
        """Fast girth check using BFS from vertex 0 on circulant graph."""
        # Precompute adjacency offsets
        offsets = []
        for s in steps:
            offsets.append(s)
            offsets.append(-s)
        
        # BFS from 0 to depth g//2 - 1
        depth = [-1] * n
        parent = [-1] * n
        depth[0] = 0
        parent[0] = -2  # root marker
        
        queue = [0]
        head = 0
        min_cycle = float('inf')
        
        # Max depth we need to explore
        max_depth = (g - 1) // 2
        
        while head < len(queue):
            u = queue[head]
            head += 1
            
            if depth[u] >= max_depth:
                continue  # No need to expand further
            
            for off in offsets:
                v = (u + off) % n
                
                if depth[v] == -1:
                    depth[v] = depth[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    # Back edge or cross edge
                    if parent[u] == v:
                        continue
                    cycle_len = depth[u] + depth[v] + 1
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
                    if min_cycle < g:
                        return min_cycle
        
        return min_cycle

    # Iterate n
    found = False
    for n in range(244, 4000, 2):
        if time.time() > deadline:
            break
        
        # Number of random samples: more for small n, fewer for large n
        if n < 400:
            num_samples = 20000
        elif n < 800:
            num_samples = 10000
        elif n < 1600:
            num_samples = 5000
        else:
            num_samples = 2000
        
        for _ in range(num_samples):
            if time.time() > deadline:
                break
            
            steps = random.sample(range(1, n // 2 + 1), 4)
            
            girth = check_girth_fast(n, steps, g)
            if girth >= g:
                # Construct edges
                edges = []
                for u in range(n):
                    for s in steps:
                        v = (u + s) % n
                        if u < v:
                            edges.append((u, v))
                return n, edges
        
    # If no circulant found, return LUW (seed behavior)
    # Re-implement seed logic for fallback
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
