import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for fallback and cleanup
    
    # Moore bound for k-regular graph of girth g
    if g % 2 == 0:
        m = g // 2
        n_min = 2 * sum((k-1)**i for i in range(m))
    else:
        m = g // 2
        n_min = (k-1)**m + k * sum((k-1)**i for i in range(m))
        
    if (n_min * k) % 2 != 0:
        n_min += 1
        
    max_n = 320
    
    # We will try to find a graph with n close to n_min.
    # If we fail, we use the fallback.
    
    for n in range(n_min, max_n + 1):
        if time.time() > deadline - 10:  # Stop trying new n if close to deadline
            break
            
        # Try multiple random restarts for this n
        # Limit restarts to avoid infinite loops
        max_restarts = 50 
        for r in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            adj = [set() for _ in range(n)]
            deg = [0] * n
            success = True
            
            while True:
                if time.time() > deadline - 2:
                    success = False
                    break
                    
                candidates = [i for i in range(n) if deg[i] < k]
                if not candidates:
                    break # All degrees are k
                    
                # Choose vertex with most remaining degree (or random among ties)
                # This can help avoid getting stuck
                max_deg_remaining = max(k - deg[i] for i in candidates)
                top_candidates = [i for i in candidates if k - deg[i] == max_deg_remaining]
                u = random.choice(top_candidates)
                
                # Efficient distance calculation: BFS from u once
                # dist[v] will be the shortest distance from u to v
                # If dist[v] >= g-1, then adding edge (u,v) creates a cycle of length >= g
                
                dist = [-1] * n
                dist[u] = 0
                queue = deque([u])
                # BFS up to depth g-2
                while queue:
                    curr = queue.popleft()
                    d_curr = dist[curr]
                    if d_curr >= g - 2:
                        continue
                    for v in adj[curr]:
                        if dist[v] == -1:
                            dist[v] = d_curr + 1
                            if dist[v] < g - 1:
                                queue.append(v)
                
                valid_vs = []
                for v in candidates:
                    if v == u:
                        continue
                    if v in adj[u]:
                        continue
                    if dist[v] == -1: # Not reached within g-2
                        valid_vs.append(v)
                        
                if not valid_vs:
                    success = False
                    break
                else:
                    # Choose v randomly, but perhaps prefer those with higher remaining degree
                    # to balance the construction
                    v = random.choice(valid_vs)
                    adj[u].add(v)
                    adj[v].add(u)
                    deg[u] += 1
                    deg[v] += 1
                    
            if success:
                # Verify
                for i in range(n):
                    if len(adj[i]) != k:
                        success = False
                        break
                if success:
                    edges = []
                    for i in range(n):
                        for j in adj[i]:
                            if i < j:
                                edges.append((i, j))
                    return n, edges

    # Fallback: LUW graph
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
