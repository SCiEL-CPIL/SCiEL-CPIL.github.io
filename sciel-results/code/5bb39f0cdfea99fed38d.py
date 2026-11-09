import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    # Moore bound for n(k,g)
    # For odd g: n >= 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i
    if g % 2 == 1:
        moore = 1 + k * sum((k-1)**i for i in range((g-3)//2 + 1))
    else:
        moore = 2 * sum((k-1)**i for i in range(g//2))
    
    # Try randomized search
    best_n = float('inf')
    best_edges = None
    
    # Try different n values, starting from Moore bound
    n_start = max(moore, 50)
    
    # Try a few random graphs
    for n in range(n_start, n_start + 200):  # Try up to 200 different n values
        if time.time() > deadline - 30:
            break
        
        # Generate random k-regular graph using configuration model
        for attempt in range(10):
            if time.time() > deadline - 30:
                break
            
            # Configuration model
            half_edges = k * n // 2
            if k * n % 2 != 0:
                continue
            
            # Create stubs
            stubs = []
            for i in range(n):
                stubs.extend([i] * k)
            
            random.shuffle(stubs)
            
            # Pair stubs to form edges
            edges = set()
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    valid = False
                    break
                if (min(u,v), max(u,v)) in edges:
                    valid = False
                    break
                edges.add((min(u,v), max(u,v)))
            
            if not valid:
                continue
            
            # Check if graph is connected
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            # BFS to check connectivity
            visited = [False] * n
            queue = deque([0])
            visited[0] = True
            count = 0
            while queue:
                node = queue.popleft()
                count += 1
                for neighbor in adj[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            
            if count != n:
                continue
            
            # Check girth
            girth = n + 1
            for start in range(n):
                if time.time() > deadline - 30:
                    break
                # BFS to find shortest cycle
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                queue = deque([start])
                
                while queue:
                    u = queue.popleft()
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue.append(v)
                        elif v != parent[u] and parent[v] != u:
                            # Found a cycle
                            cycle_len = dist[u] + dist[v] + 1
                            girth = min(girth, cycle_len)
                
                if girth < g:
                    break
            
            if girth >= g and n < best_n:
                best_n = n
                best_edges = list(edges)
                break
        
        if best_n == n:
            break
    
    if best_edges is not None:
        return best_n, best_edges
    
    # Fallback: L-U-W construction
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
