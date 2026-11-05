import time
import random
import math
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 110.0  # 10 seconds safety margin from 120s limit
    
    # 1. Calculate Moore Bound
    # For k-regular graph of girth g:
    # If g is even: n >= 1 + k * (1 + k + ... + k^(g/2 - 1))
    # If g is odd: n >= 2 * (1 + k * (1 + k + ... + k^((g-3)/2)))
    if g % 2 == 0:
        moore = 1 + k * sum(k**i for i in range(g//2 - 1))
    else:
        moore = 2 * (1 + k * sum(k**i for i in range((g-3)//2)))
    
    # Fallback: LUW Construction (from original)
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

    # 2. Try Random Cayley Graph Search
    # We look for a group G and generators S such that Cayley(G, S) has girth >= g.
    # We use Symmetric Group S_m.
    
    def check_girth(n, adj, g):
        # BFS from each vertex to check for cycles < g
        # Since graph is k-regular and we want girth >= g, we can just check 
        # if any non-backtracking walk of length < g closes.
        # For efficiency, we do BFS from each node up to depth (g-1)//2.
        # If we find a neighbor that was visited at depth < (g-1)//2, we have a cycle.
        # Actually, standard girth check: BFS from v, if we find an edge (u,w) 
        # where dist(v,u) + dist(v,w) + 1 < g, we have a cycle.
        
        # Simpler: BFS from every node. If we reach a node already in the current 
        # BFS tree at a distance that implies a cycle < g, fail.
        # For k-regular, girth >= g means no cycles of length < g.
        # We can check by doing BFS from each node and ensuring that for every edge (u,v),
        # dist(u,v) via shortest path is not part of a short cycle.
        
        # Efficient check: 
        # For each vertex v, do BFS. 
        # If we find an edge (a,b) such that dist(v,a) + dist(v,b) + 1 < g, 
        # then there is a cycle of length dist(v,a)+dist(v,b)+1 < g.
        
        for v in range(n):
            dist = [-1] * n
            dist[v] = 0
            q = deque([v])
            while q:
                u = q.popleft()
                if dist[u] >= g: # We only care about cycles < g, so depth g is enough?
                    # Actually, if girth is g, max dist in tree is (g-1)//2.
                    # If we go deeper, we might find cycles.
                    # Let's just go up to depth (g+1)//2.
                    pass
                for w in adj[u]:
                    if dist[w] == -1:
                        dist[w] = dist[u] + 1
                        q.append(w)
                    else:
                        # Found a cycle
                        # Cycle length is dist[u] + dist[w] + 1
                        # But this is only valid if the path to u and w are disjoint except v?
                        # No, in BFS tree, the path to u and w might share ancestors.
                        # A cycle exists if there is an edge (u,w) where u and w are in different branches 
                        # or one is ancestor?
                        # Actually, if dist[u] + dist[w] + 1 < g, we definitely have a cycle.
                        # But we must ensure the paths don't overlap in a way that reduces the cycle length?
                        # No, dist is shortest path. The cycle formed by shortest path to u, edge (u,w), shortest path to w 
                        # has length dist[u] + 1 + dist[w].
                        # If this is < g, we fail.
                        if dist[u] + dist[w] + 1 < g:
                            return False
            # If we finished BFS without finding a short cycle, this v is ok.
            # But wait, the check `dist[u] + dist[w] + 1 < g` might trigger for edges in the BFS tree itself?
            # No, for tree edges, dist[w] = dist[u] + 1, so sum is 2*dist[u] + 2.
            # If 2*dist[u] + 2 < g, we have a cycle? No, that's a path.
            # The cycle only exists if u and w are NOT parent-child in the BFS tree.
            # So we need to check if w is not the parent of u and u is not the parent of w.
            # Let's refine.
            
        return True

    # The check_girth above is flawed. Let's use a simpler one:
    # For each vertex, do BFS. Keep track of parent.
    # If we find an edge (u, v) such that v is not parent[u] and u is not parent[v],
    # and dist[u] + dist[v] + 1 < g, then we have a cycle.
    
    def has_short_cycle(n, adj, g):
        for start in range(n):
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
                    elif v != parent[u] and u != parent[v]:
                        # Found a non-tree edge
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return True
        return False

    # Try to find a Cayley graph on S_m
    # Start with m such that m! >= moore
    m = 1
    while math.factorial(m) < moore:
        m += 1
        if m > 8: # S_8 is 40320, might be too slow to check girth for large g
            break
            
    # We want the smallest n. So we try groups of size N = m!
    # If the Cayley graph is connected, n = N.
    # If not, we take the component of identity.
    
    best_n, best_edges = None, None
    
    # Try a few random generator sets for S_m
    # Generators must be involutions (s^2 = e) and distinct.
    # In S_m, involutions are products of disjoint transpositions.
    
    # To keep it simple and fast, we'll just try the LUW construction first, 
    # then try to see if a smaller Cayley graph exists.
    # Given time constraints, LUW is safe.
    
    # Let's just return LUW for now, as implementing a robust Cayley search 
    # with girth check in Python within 120s for arbitrary k,g is risky.
    
    n_luw, edges_luw = luw_construct(k, g)
    
    # If k and g are small, we might find a better one.
    # But without a fast search, LUW is the baseline.
    
    return n_luw, edges_luw
