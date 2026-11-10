import random
import time
from collections import deque
import numpy as np

def construct(k, g):
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def luw_construct(k, g):
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

    # Moore bound
    if g % 2 == 0:
        r = g // 2
        moore = 2 * sum((k-1)**i for i in range(r))
    else:
        r = (g-1)//2
        moore = 1 + k * sum((k-1)**i for i in range(r))
        
    start_time = time.time()
    timeout = 110 # Leave some time for fallback
    
    # New Approach: Random Regular Graph Search
    # We search for n from moore to moore + 200
    # Since girth is even (10), the graph is likely bipartite, so n should be even.
    
    best_n = None
    best_edges = None
    
    # Step 1: Try to find a random k-regular graph with girth >= g
    # We iterate n from moore upwards.
    
    max_n_to_try = moore + 200
    
    for n in range(moore, max_n_to_try + 1):
        if time.time() - start_time > timeout:
            break
            
        # If g is even, girth is even, graph must be bipartite?
        # No, girth 10 doesn't force bipartite, but cages with even girth are often bipartite.
        # However, a graph with girth 10 can have odd cycles?
        # If it has an odd cycle, its length is at least 11 (if girth is 10, no cycles < 10).
        # So odd cycles can exist.
        # But the (4,10)-cage is bipartite.
        # Let's try both even and odd n, but prioritize even.
        
        # Try a few random graphs for this n
        # Number of attempts depends on remaining time
        attempts = 500 if n < moore + 50 else 100
        
        for _ in range(attempts):
            if time.time() - start_time > timeout:
                break
                
            # Generate random k-regular graph using Configuration Model
            # Half-edges: n vertices, k stubs each.
            # Total stubs: n * k
            # Pair them randomly.
            
            stubs = [i for i in range(n) for _ in range(k)]
            random.shuffle(stubs)
            
            # Pair up
            edges_set = set()
            valid = True
            for i in range(0, len(stubs), 2):
                u = stubs[i]
                v = stubs[i+1]
                if u == v:
                    valid = False
                    break
                if u > v:
                    u, v = v, u
                if (u, v) in edges_set:
                    valid = False
                    break
                edges_set.add((u, v))
            
            if not valid:
                continue
                
            # Check degree (should be k by construction, but verify no multi-edges handled above)
            # Check girth
            # Build adjacency list
            adj = [[] for _ in range(n)]
            for u, v in edges_set:
                adj[u].append(v)
                adj[v].append(u)
                
            # Verify degree
            deg_ok = True
            for i in range(n):
                if len(adj[i]) != k:
                    deg_ok = False
                    break
            if not deg_ok:
                continue
                
            # Check girth
            # BFS from each vertex? Too slow.
            # BFS from one vertex? Not sufficient for non-transitive.
            # We need to check if ANY cycle < g exists.
            # We can run BFS from all vertices, but prune if we find a short cycle.
            # Or, just run BFS from vertex 0. If it finds a cycle < g, reject.
            # If it doesn't, we still need to check other vertices.
            # However, if the graph is random, it's likely that if a short cycle exists, it's "visible" from many vertices.
            # But to be safe, we must check all.
            # Optimization: If n is small, check all.
            
            girth_ok = True
            for start in range(n):
                # BFS
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                q = deque([start])
                while q:
                    u = q.popleft()
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            q.append(v)
                        elif parent[u] != v and parent[v] != u:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                girth_ok = False
                                break
                    if not girth_ok:
                        break
                if not girth_ok:
                    break
                    
            if girth_ok:
                # Found a valid graph
                return n, list(edges_set)
                
    # Fallback
    return luw_construct(k, g)
