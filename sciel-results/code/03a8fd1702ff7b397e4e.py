import time
import math
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    def luw_construction(d, q):
        """
        Constructs the LUW graph for given d and q.
        Returns (n, edges).
        """
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
        
        # Rule definition
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
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
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
        
        return len(order), edges

    def get_girth_fast(n, edges, target_girth):
        """
        Checks if the graph has girth at least target_girth.
        Returns True if girth >= target_girth, else False.
        Uses BFS from each vertex up to depth target_girth//2.
        """
        if n == 0:
            return True
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        max_depth = target_girth // 2
        # For odd girth, we need to be careful.
        # If girth is g, then no cycle of length < g.
        # BFS from u: if we find a neighbor v such that dist[u->v] < g, and v is not parent...
        # Standard check: BFS from u. If we encounter a visited node v at depth d_v 
        # from u, and the path length is d_u + d_v + 1.
        # We just need to ensure no such sum < g.
        
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        # Check if this edge closes a short cycle
                        # The cycle length is dist[u] + dist[v] + 1
                        # But only if u and v are not parent-child in the BFS tree
                        # If v is parent of u, dist[v] = dist[u] - 1, cycle length 2 (back edge)
                        # We ignore back edges to parents.
                        if parent[u] != v and parent[v] != u:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < target_girth:
                                return False
            if time.time() > deadline:
                return False
        return True

    # Determine candidate d values
    q = k
    
    min_d = max(1, (g - 4 + 1) // 2)
    
    # We want to try d values that yield small n.
    # n = 2 * q^d
    # Try d starting from min_d upwards
    best_result = None
    best_n = float('inf')
    
    # Limit d to avoid huge graphs
    max_d = 10
    
    for d in range(min_d, max_d + 1):
        if time.time() > deadline - 10:
            break
            
        try:
            n, edges = luw_construction(d, q)
        except Exception:
            continue
            
        # Check if it's k-regular
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if any(dg != k for dg in deg):
            continue
            
        # Check girth
        if n < 10000:
            if not get_girth_fast(n, edges, g):
                continue
        else:
            # Trust the theoretical bound
            if 2 * d + 4 < g:
                continue
                
        if n < best_n:
            best_n = n
            best_result = (n, edges)
            # If we found a valid graph, we can break if we are confident it's the smallest
            # Since n grows with d, the first valid d is the smallest n
            break
            
    if best_result:
        return best_result
    else:
        # Fallback to a larger d if none found
        d = min_d + 3
        return luw_construction(d, q)
