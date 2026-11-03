import time
import sys
import collections
import math

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    if k != 3:
        # Fallback for other k: use LUBW if possible, or random
        # For now, focus on k=3
        pass

    # Strategy:
    # 1. Search for Generalized Petersen Graphs GP(n, k_gp) with girth >= g.
    # 2. Search for Circulant Graphs C_n({a, -a, n/2}) with girth >= g.
    # 3. Fallback to LUW construction (known to have girth >= 16 for d=10, q=3).

    best_n = float('inf')
    best_edges = None

    # Helper: Check girth using BFS
    def check_girth(n, edges, min_girth):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        max_depth = min_girth // 2 - 1
        
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = collections.deque([start])
            
            while queue:
                curr = queue.popleft()
                if dist[curr] > max_depth:
                    continue
                for neighbor in adj[curr]:
                    if dist[neighbor] == -1:
                        dist[neighbor] = dist[curr] + 1
                        parent[neighbor] = curr
                        queue.append(neighbor)
                    elif neighbor != parent[curr]:
                        # Found a cycle
                        if dist[curr] + dist[neighbor] + 1 < min_girth:
                            return False
        return True

    # Search GP Graphs
    # n is the number of vertices in the outer cycle. Total vertices = 2n.
    # We want 2n to be small.
    # Moore bound is 512, so 2n >= 512 => n >= 256.
    # Known best is 936 => n = 468.
    # We search n from 256 to 600.
    
    max_n_search = 600
    for n in range(256, max_n_search + 1):
        if time.time() > deadline - 20.0:
            break
        
        for k_gp in range(1, n // 2 + 1):
            if time.time() > deadline - 15.0:
                break
            
            if math.gcd(n, k_gp) != 1:
                continue
            
            edges_gp = []
            # Outer cycle
            for i in range(n):
                edges_gp.append((i, (i + 1) % n))
            # Spokes
            for i in range(n):
                edges_gp.append((i, n + i))
            # Inner cycle
            for i in range(n):
                edges_gp.append((n + i, n + (i + k_gp) % n))
            
            if check_girth(2 * n, edges_gp, g):
                return 2 * n, edges_gp

    # Search Circulant Graphs
    # C_n({a, -a, n/2})
    # n must be even.
    # Vertices 0..n-1.
    # Edges: (i, i+a), (i, i-a), (i, i+n/2) mod n.
    
    for n in range(512, 1000, 2): # Start from Moore bound, even
        if time.time() > deadline - 10.0:
            break
        
        for a in range(1, n // 2):
            if time.time() > deadline - 5.0:
                break
            
            # Avoid duplicate edges/loops
            if a == 0 or a == n // 2:
                continue
            if 2 * a % n == 0: # a = n/2, already handled
                continue
            if a == n - a: # 2a = n, handled
                continue
            
            edges_circ = []
            for i in range(n):
                # Edge to i+a
                edges_circ.append((i, (i + a) % n))
                # Edge to i+n/2
                edges_circ.append((i, (i + n // 2) % n))
                # Edge to i-a is the reverse of (i-a, i+a) from previous iteration?
                # No, we add undirected edges.
                # (i, i+a) covers (i+a, i).
                # We need (i, i-a). This is (i, i+n-a).
                # Is (i, i+n-a) the same as (i-a, i+a)?
                # Let j = i-a. Edge (j, j+a) = (i-a, i).
                # So iterating i from 0 to n-1 and adding (i, i+a) covers all "a" edges.
                # We just need to be careful not to double count.
                # Since it's a set of edges, we can use a set or just add and deduplicate.
                # For speed, we assume the structure is simple.
                # Actually, adding (i, i+a) for all i adds each edge twice (once from i, once from i+a).
                # So we can just add (i, (i+a)%n) for i in 0..n-1.
                # And (i, (i+n//2)%n) for i in 0..n-1.
                # This creates 2n edges? No, n edges of type A, n edges of type B.
                # Total 2n edges? No, each (i, j) is one edge.
                # Type A: n edges.
                # Type B: n edges.
                # Total 2n edges.
                # But graph is 3-regular, so edges should be 3n/2.
                # My construction adds (i, i+a) and (i, i-a)?
                # If I only add (i, i+a), I get n edges.
                # I need (i, i-a) as well.
                # (i, i-a) is the same undirected edge as (i-a, i).
                # So adding (i, (i+a)%n) for all i covers all "a" connections.
                # So Type A contributes n edges.
                # Type B: (i, i+n/2). This is an involution.
                # (i, i+n/2) and (i+n/2, i+n/2+n/2=i) are the same edge.
                # So adding (i, i+n/2) for all i adds each edge twice.
                # So Type B contributes n/2 unique edges.
                # Total edges: n + n/2 = 1.5n. Correct.
                
                # To avoid duplicates in the list, we can just generate and let the evaluator handle it?
                # No, "list of edges". Duplicates might be invalid? "no repeated edges".
                # So we must deduplicate.
                # For Type A: (i, i+a). If we iterate i, we get (0, a), (1, a+1)...
                # These are distinct.
                # For Type B: (i, i+n/2). (0, n/2), (1, n/2+1)...
                # (n/2, 0) is the same as (0, n/2).
                # So we only iterate i from 0 to n/2 - 1 for Type B.
                
                pass # Logic inside loop below

            # Re-implementing edge generation cleanly
            edges_circ = []
            # Type A: (i, i+a)
            for i in range(n):
                edges_circ.append((i, (i + a) % n))
            # Type B: (i, i+n/2) for i in 0..n/2-1
            for i in range(n // 2):
                edges_circ.append((i, i + n // 2))
            
            if check_girth(n, edges_circ, g):
                return n, edges_circ

    # Fallback: LUW
    # q=3, d=10 (girth 16)
    q = 3
    d = 10
    
    def add(x, y): return (x + y) % q
    def sub(x, y): return (x - y) % q
    def mul(x, y): return x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rules_cycle = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = rules_cycle[r]

    index = {}
    order = []
    edges = []
    
    start = (0, (None,) + (0,) * d)
    index[start] = 0
    order.append(start)
    
    for i, vertex in enumerate(order):
        if time.time() > deadline - 1.0:
            break
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            
            neighbor = (1 - side, tuple(y))
            
            if neighbor not in index:
                index[neighbor] = len(order)
                order.append(neighbor)
            
            if side == 0:
                edges.append((index[vertex], index[neighbor]))
                
    n_luw = len(order)
    return n_luw, edges
