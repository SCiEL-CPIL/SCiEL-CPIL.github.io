"""Deterministic algebraic construction for regular graphs of large girth."""
import time
import random

# EVOLVE-BLOCK-START
def construct(k, g):
    t0 = time.time()
    deadline = 120.0  # seconds; evaluator kills at 150s, keep margin

    # Use the LUW-type algebraic construction for all cases
    q = k
    # Choose depth d such that the resulting graph has girth >= g
    # The LUW construction gives girth approximately 2*d + 1 or 2*d + 2
    # We need girth >= g, so we need d such that 2*d + 1 >= g (for odd girth)
    # or 2*d + 2 >= g (for even girth)
    # Conservative: use d = ceil(g/2) - 1
    d = max(3, (g + 1) // 2 - 1)
    
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add = lambda x,y: x^y
        sub = lambda x,y: x^y
        mul = lambda x,y: table[x][y]
    else:
        add = lambda x,y: (x+y)%q
        sub = lambda x,y: (x-y)%q
        mul = lambda x,y: x*y%q
    
    # Define the LUW rule for each column
    rule = {2:(1,1), 3:(2,1), 4:(1,2)}
    for c in range(5, d+1):
        r, c0 = (c-5)%4, c-(c-5)%4
        rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0]*(d-1)
            for c in range(2, d+1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
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
    
    # Verify girth using non-backtracking walk enumeration
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check girth >= g using BFS with parent tracking (non-backtracking)
    def verify_girth(n, adj, min_girth):
        max_dist = (min_girth - 1) // 2
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            qi = 0
            while qi < len(queue):
                u = queue[qi]
                qi += 1
                if dist[u] >= max_dist:
                    continue
                du = dist[u]
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
                        cycle_len = du + dv + 1
                        if cycle_len < min_girth:
                            return False
        return True
    
    if not verify_girth(n, adj, g):
        # If the construction doesn't meet girth, try with larger d
        for d_try in range(d+1, d+5):
            if time.time() - t0 > deadline - 10:
                break
            rule2 = {2:(1,1), 3:(2,1), 4:(1,2)}
            for c in range(5, d_try+1):
                r, c0 = (c-5)%4, c-(c-5)%4
                rule2[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
            
            def neighbours2(vertex):
                side, x = vertex
                for first in range(q):
                    y = [None, first] + [0]*(d_try-1)
                    for c in range(2, d_try+1):
                        a, b = rule2[c]
                        y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                    yield 1-side, tuple(y)
            
            start2 = (0, (None,) + (0,)*d_try)
            index2 = {start2: 0}
            order2 = [start2]
            edges2 = []
            for vertex in order2:
                for other in neighbours2(vertex):
                    if other not in index2:
                        index2[other] = len(order2)
                        order2.append(other)
                    if vertex[0] == 0:
                        edges2.append((index2[vertex], index2[other]))
            
            n2 = len(order2)
            adj2 = [[] for _ in range(n2)]
            for u, v in edges2:
                adj2[u].append(v)
                adj2[v].append(u)
            
            if verify_girth(n2, adj2, g):
                return n2, edges2
    
    return n, edges
# EVOLVE-BLOCK-END
