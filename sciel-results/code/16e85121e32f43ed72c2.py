import time
import sys
from collections import deque

def construct(k, g):
    t0 = time.time()
    deadline = 115.0  # seconds; keep margin before 150s kill

    # The LUW construction works for q = k if k is a prime power.
    # For k=3, q=3.
    q = k
    
    if q == 4:
        # Specific table for GF(4)
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add = lambda x,y: x^y
        sub = lambda x,y: x^y
        mul = lambda x,y: table[x][y]
    else:
        # Field arithmetic mod q (assumes q is prime)
        add = lambda x,y: (x+y)%q
        sub = lambda x,y: (x-y)%q
        mul = lambda x,y: x*y%q

    def build_rule(d):
        rule = {2:(1,1), 3:(2,1), 4:(1,2)}
        for c in range(5, d+1):
            r, c0 = (c-5)%4, c-(c-5)%4
            rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
        return rule

    def build_graph(d, rule):
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1-side, tuple(y)

        start = (0, (None,) + (0,)*d)
        index = {start: 0}
        order = [start]
        edges = []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))

        n = len(order)
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return n, edges, adj

    def verify_girth(n, adj, min_girth):
        """
        Check girth >= min_girth.
        We use BFS from each vertex. If we find a back-edge to an ancestor 
        at a distance creating a cycle < min_girth, return False.
        """
        max_dist = (min_girth - 1) // 2
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                du = dist[u]
                if du >= max_dist:
                    continue
                pu = parent[u]
                for v in adj[u]:
                    if v == pu:
                        continue
                    dv = dist[v]
                    if dv == -1:
                        dist[v] = du + 1
                        parent[v] = u
                        queue.append(v)
                    elif dv >= du:
                        # Cycle found: u -> ... -> v -> u
                        # Length is du + dv + 1
                        cycle_len = du + dv + 1
                        if cycle_len < min_girth:
                            return False
        return True

    # Theoretical minimum d for girth g is ceil((g-1)/2)
    # But for LUW, girth is often 2d+2.
    # For g=16, 2d+2 >= 16 => d >= 7.
    # So we should try d=7 first.
    
    d_min = (g - 1) // 2  # 7 for g=16
    
    best_result = None
    last_valid_n = None
    last_valid_edges = None

    # Try increasing d
    # Upper bound on d: if d gets too large, n explodes.
    # For k=3, n = 2*3^d.
    # d=8 -> 13122.
    # d=7 -> 4374.
    
    max_d = 10  # Safety cap

    for d in range(d_min, max_d + 1):
        if time.time() - t0 > deadline - 5.0:
            break
            
        rule = build_rule(d)
        try:
            n, edges, adj = build_graph(d, rule)
        except MemoryError:
            break
        except Exception:
            continue

        # Check girth
        if verify_girth(n, adj, g):
            return n, edges
        
    # If we failed to find a graph with d < max_d, return the last one constructed?
    # No, we must return a valid graph.
    # If the loop finishes without return, it means we didn't find a valid girth.
    # This is unlikely for d=7.
    
    return 0, []

# EVOLVE-BLOCK-END
