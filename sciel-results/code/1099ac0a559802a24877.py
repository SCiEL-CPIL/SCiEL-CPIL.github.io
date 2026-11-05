import time
import random
import math
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 145.0 # Safety margin

    # Fallback: LUW Construction (guaranteed valid, potentially large)
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

    if k != 3:
        return luw_construct(k, g)

    # Target n for k=3, g=16
    n = 936
    
    # Generate random 3-regular graph
    def gen_3reg(n):
        stubs = [i for i in range(n) for _ in range(3)]
        while True:
            random.shuffle(stubs)
            edges = set()
            adj = [[] for _ in range(n)]
            ok = True
            for i in range(0, 3*n, 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    ok = False; break
                if (u, v) in edges or (v, u) in edges:
                    ok = False; break
                edges.add((u, v))
                adj[u].append(v)
                adj[v].append(u)
            if ok:
                return adj, list(edges)

    # Check if graph has girth >= g
    # Efficient check: For each edge (u,v), check if dist(u,v) < g-1 in G - (u,v)
    # If yes, there is a cycle of length dist(u,v) + 1 < g
    def has_short_cycle(adj, n, g):
        # Build adjacency set for fast lookup
        adj_set = [set(adj[i]) for i in range(n)]
        for u in range(n):
            for v in adj[u]:
                if v < u: continue # Check each edge once
                # Remove u-v temporarily
                adj_set[u].remove(v)
                adj_set[v].remove(u)
                
                # BFS from u to v
                dist = [-1] * n
                dist[u] = 0
                q = deque([u])
                found = False
                while q:
                    curr = q.popleft()
                    if dist[curr] >= g - 1: # We only care if dist < g-1
                        continue
                    for nb in adj_set[curr]:
                        if dist[nb] == -1:
                            dist[nb] = dist[curr] + 1
                            if nb == v:
                                found = True
                                break
                            q.append(nb)
                    if found:
                        break
                if found:
                    # Restore and return True
                    adj_set[u].add(v)
                    adj_set[v].add(u)
                    return True
                
                # Restore
                adj_set[u].add(v)
                adj_set[v].add(u)
        return False

    # Simulated Annealing
    best_adj = None
    best_edges = None
    
    for restart in range(10):
        if time.time() > deadline - 5: break
        
        adj, edges = gen_3reg(n)
        
        # Check initial girth
        if not has_short_cycle(adj, n, g):
            return n, edges
            
        T = 1.0
        # We don't track exact energy, we just accept if it reduces short cycles or with prob
        # To be fast, we only check local changes.
        
        for step in range(20000):
            if time.time() > deadline - 5: break
            
            # Pick two random edges
            e1 = random.choice(edges)
            u1, v1 = e1
            e2 = random.choice(edges)
            u2, v2 = e2
            
            if e1 == e2: continue
            if u1 in [u2, v2] or v1 in [u2, v2]:
                continue
                
            # Try swap (u1, u2) and (v1, v2)
            # Check if u1==u2 or v1==v2 (loops)
            if u1 == u2 or v1 == v2: continue
            if u1 == v2 or v1 == u2: continue # Multi-edge?
            
            # Check if edges already exist
            if u2 in adj[u1] or v2 in adj[v1]: continue
            if v2 in adj[u1] or u2 in adj[v1]: continue # Wait, we are adding u1-u2 and v1-v2
            
            # Perform swap
            adj[u1].remove(v1)
            adj[v1].remove(u1)
            adj[u2].remove(v2)
            adj[v2].remove(u2)
            
            adj[u1].append(u2)
            adj[u2].append(u1)
            adj[v1].append(v2)
            adj[v2].append(v1)
            
            edges.remove(e1)
            edges.remove(e2)
            edges.append((u1, u2))
            edges.append((v1, v2))
            
            # Check if this created a short cycle
            # Only need to check cycles involving new edges
            # But has_short_cycle is O(n * E) which is too slow for every step.
            # We need a faster check.
            # For now, let's just run has_short_cycle every 100 steps?
            # Or just accept and check at the end?
            # If we accept bad moves, we might get stuck.
            
            # Let's do a quick local check:
            # Is dist(u1, u2) < g-1 in the new graph?
            # This is still BFS.
            
            # Given time constraints, we'll just do a full check periodically.
            if step % 100 == 0:
                if not has_short_cycle(adj, n, g):
                    return n, edges
                
            # Cooling
            T *= 0.999
            
    # If SA failed, return LUW
    return luw_construct(k, g)
