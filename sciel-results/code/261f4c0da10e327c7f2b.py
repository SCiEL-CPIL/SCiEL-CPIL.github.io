import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Moore Bound
    if k == 2:
        n_min = g
    else:
        # 2 + k * sum_{i=0}^{g/2 - 2} (k-1)^i
        if g % 2 == 0:
            # Diameter g/2 - 1
            d = g // 2 - 1
            n_min = 2 + k * (( (k-1)**(d+1) - 1 ) // (k-2)) if k > 2 else 2 + k*(d+1)
        else:
            # Diameter (g-1)/2
            d = (g - 1) // 2
            n_min = 1 + k * (( (k-1)**d - 1 ) // (k-2)) if k > 2 else 1 + k*d
            
    # For k=3, g=16:
    # d = 7. n_min = 2 + 3 * (3^8 - 1)/2 = 2 + 3 * (6561-1)/2 = 2 + 3*3280 = 9842? 
    # Wait. Moore bound for (3,16) is 512.
    # Formula: 2 + k * sum_{i=0}^{d-1} (k-1)^i where d = g/2 - 1.
    # For g=16, d=7. Sum i=0 to 6 of 2^i = 2^7 - 1 = 127.
    # n_min = 2 + 3 * 127 = 2 + 381 = 383?
    # Let's re-verify Moore bound for (3,16).
    # Diameter is g/2 - 1 = 7.
    # Tree of depth 7 from a node has 1 + 3 + 6 + 12 + 24 + 48 + 96 + 192 = 382 nodes.
    # So n >= 383?
    # Actually, the Moore bound is $2 + k \sum_{i=0}^{g/2-2} (k-1)^i$ for even g.
    # g=16 -> g/2 - 2 = 6.
    # Sum i=0 to 6 of 2^i = 127.
    # 2 + 3*127 = 383.
    # Why did the prompt say 512?
    # "best lower bound is 512". The Moore bound is a lower bound, but often not tight.
    # Let's use 512 as the starting point to be safe, or calculate it.
    
    # Let's use the prompt's lower bound 512 for k=3, g=16 as a heuristic start, 
    # but generally calculate Moore bound.
    
    def get_moore_bound(k, g):
        if k == 2:
            return g
        if g % 2 == 0:
            d = g // 2 - 1
            if k == 3:
                # Sum 2^i for i=0 to d-1
                s = 2**d - 1
                return 2 + 3 * s
            else:
                s = ((k-1)**d - 1) // (k-2)
                return 2 + k * s
        else:
            d = (g - 1) // 2
            if k == 3:
                s = 2**d - 1
                return 1 + 3 * s
            else:
                s = ((k-1)**d - 1) // (k-2)
                return 1 + k * s

    n_start = get_moore_bound(k, g)
    # The prompt says lower bound 512 for (3,16). My calc: d=7. 2 + 3*(2^7-1) = 2 + 381 = 383.
    # Maybe the diameter is different?
    # Girth 16 means no cycles < 16.
    # The "cage" is the smallest graph.
    # Let's start a bit higher to increase probability of success, e.g., 512 or 600.
    
    # Strategy:
    # 1. Try random graphs for n in range [n_start, max_n].
    # 2. If found, return.
    # 3. If time runs out, return LUW graph (valid but large).
    
    def generate_random_regular_graph(n, k):
        # Configuration model
        # 1. Create list of half edges
        # 2. Shuffle
        # 3. Pair
        # 4. Check simplicity
        
        if n * k % 2 != 0:
            return None
            
        # For efficiency, use numpy
        half_edges = np.repeat(np.arange(n), k)
        np.random.shuffle(half_edges)
        
        # Pair them
        edges = []
        for i in range(0, len(half_edges), 2):
            u = half_edges[i]
            v = half_edges[i+1]
            if u == v:
                return None # Loop
            edges.append((u, v))
            
        # Check for multi-edges
        edge_set = set()
        for u, v in edges:
            key = (min(u, v), max(u, v))
            if key in edge_set:
                return None
            edge_set.add(key)
            
        return edges

    def check_girth(n, edges, g):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # BFS from each node to depth g/2 - 1
        max_depth = g // 2 - 1
        
        for s in range(n):
            # dist[v] = shortest distance from s to v
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
                    else:
                        # Found a cycle
                        # Cycle length = dist[u] + dist[v] + 1
                        # But dist[v] might be from a different branch.
                        # If dist[u] + dist[v] + 1 < g, we have a short cycle.
                        # Note: This check might detect the same cycle multiple times, 
                        # but if any cycle < g is found, we fail.
                        # However, dist[v] is the *shortest* distance.
                        # If we encounter an edge (u,v) where both are visited,
                        # the cycle length is dist[u] + dist[v] + 1.
                        # If this is < g, return False.
                        if dist[u] + dist[v] + 1 < g:
                            return False
                            
        return True

    # LUW Fallback
    def luw_construct(k, g):
        # Re-use the logic from the seed program
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

    # Main Loop
    # Start from Moore bound, go up in steps.
    # If k=3, g=16, start at 512 (prompt lower bound) or 400.
    # Let's start at 400 and go up by 10.
    
    n_current = max(n_start, 400)
    best_result = None
    
    while time.time() < deadline:
        # Try a few random graphs for this n
        attempts = 0
        while time.time() < deadline and attempts < 50:
            edges = generate_random_regular_graph(n_current, k)
            if edges is not None:
                if check_girth(n_current, edges, g):
                    return n_current, edges
            attempts += 1
        
        # Increase n
        n_current += 10
        
        # Safety: if n gets too big, stop random search and use LUW
        if n_current > 5000:
            break
            
    # Fallback to LUW
    return luw_construct(k, g)
