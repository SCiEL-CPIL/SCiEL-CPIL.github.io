"""Girth-constrained random regular graph via configuration model with dynamic edge rejection."""
import random
import sys
import time
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g using configuration model
    with dynamic edge rejection.
    
    Returns (n, edges) where n is number of vertices and edges is list of (i,j) pairs.
    """
    start_time = time.time()
    timeout = 115  # Leave some margin
    
    # Moore bound for (3,16): 2 * sum_{i=0}^{7} 2^i = 2*(2^8-1) = 510
    # n must be even for k=3, so n >= 512
    
    for n in range(512, 702, 2):
        if time.time() - start_time > timeout:
            break
        
        max_attempts = 40
        for attempt in range(max_attempts):
            if time.time() - start_time > timeout:
                break
            
            # Create stubs: vertex i has k stubs
            stubs = []
            for i in range(n):
                for j in range(k):
                    stubs.append((i, j))
            
            random.shuffle(stubs)
            
            adj = [[] for _ in range(n)]
            edges = []
            valid = True
            seen_edges = set()
            
            # Process stubs in pairs
            for idx in range(0, len(stubs), 2):
                u, _ = stubs[idx]
                v, _ = stubs[idx + 1]
                
                # Check for loop
                if u == v:
                    valid = False
                    break
                
                edge = (min(u, v), max(u, v))
                if edge in seen_edges:
                    valid = False
                    break
                
                # Check girth: BFS from u to see if v is reachable in < g-1 steps
                dist = [0] * n
                dist[u] = 0
                visited = [False] * n
                visited[u] = True
                dq = deque([u])
                found_short = False
                while dq:
                    cur = dq.popleft()
                    d = dist[cur]
                    if d >= g - 1:
                        break
                    for nb in adj[cur]:
                        if not visited[nb]:
                            visited[nb] = True
                            dist[nb] = d + 1
                            if nb == v:
                                found_short = True
                                break
                            dq.append(nb)
                    if found_short:
                        break
                
                if found_short:
                    valid = False
                    break
                
                seen_edges.add(edge)
                adj[u].append(v)
                adj[v].append(u)
                edges.append((u, v))
            
            if not valid:
                continue
            
            # Verify all vertices have degree k
            all_ok = True
            for i in range(n):
                if len(adj[i]) != k:
                    all_ok = False
                    break
            
            if all_ok:
                return n, edges
    
    # Fallback: use algebraic construction
    # Lazebnik-Ustimenko-Woldar type construction
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

# EVOLVE-BLOCK-END
