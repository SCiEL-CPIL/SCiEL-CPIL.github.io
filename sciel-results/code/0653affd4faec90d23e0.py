import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Calculate Moore Bound
    if g % 2 == 0:
        moore_bound = 2 * sum((k-1)**i for i in range(g//2))
    else:
        moore_bound = 1 + k * sum((k-1)**i for i in range((g-1)//2))

    best_n = None
    best_edges = None
    
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def luw_fallback():
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

    # Try random search for n from moore_bound to moore_bound + 100
    for n in range(moore_bound, moore_bound + 100, 2):
        if time.time() > deadline - 30:
            break
        
        target_edges = n * k // 2
        
        # Try multiple restarts
        for _ in range(10):
            if time.time() > deadline - 30:
                break
            
            adj = [[] for _ in range(n)]
            deg = [0] * n
            dist = np.full((n, n), np.inf)
            for i in range(n):
                dist[i][i] = 0
            
            success = True
            for step in range(target_edges):
                if time.time() > deadline - 30:
                    success = False
                    break
                
                min_deg = k
                candidates = []
                for u in range(n):
                    if deg[u] < min_deg:
                        min_deg = deg[u]
                        candidates = [u]
                    elif deg[u] == min_deg:
                        candidates.append(u)
                
                if not candidates:
                    success = False
                    break
                
                random.shuffle(candidates)
                placed = False
                for u in candidates:
                    if deg[u] >= k:
                        continue
                    
                    valid_v = []
                    for v in range(n):
                        if v == u:
                            continue
                        if deg[v] >= k:
                            continue
                        if (u, v) in adj[u]:
                            continue
                        if dist[u][v] < g - 1:
                            continue
                        valid_v.append(v)
                    
                    if not valid_v:
                        continue
                    
                    random.shuffle(valid_v)
                    for v in valid_v:
                        adj[u].append(v)
                        adj[v].append(u)
                        deg[u] += 1
                        deg[v] += 1
                        
                        new_dist_1 = dist[:, u] + 1 + dist[v, :]
                        new_dist_2 = dist[:, v] + 1 + dist[u, :]
                        np.minimum(dist, new_dist_1, out=dist)
                        np.minimum(dist, new_dist_2, out=dist)
                        
                        placed = True
                        break
                    if placed:
                        break
                
                if not placed:
                    success = False
                    break
            
            if success:
                # Check connectivity
                visited = [False] * n
                queue = [0]
                visited[0] = True
                while queue:
                    u = queue.pop(0)
                    for v in adj[u]:
                        if not visited[v]:
                            visited[v] = True
                            queue.append(v)
                if all(visited):
                    edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    best_n = n
                    best_edges = edges
                    break
        
        if best_n:
            break

    if best_n:
        return best_n, best_edges
    else:
        return luw_fallback()
