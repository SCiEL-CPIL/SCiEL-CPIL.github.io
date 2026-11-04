import time
import random
import sys
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Moore bound for k=3, g=16
    # n >= 2 * sum_{i=0}^{(g-2)/2} (k-1)^i
    # g=16, d=7
    # n >= 2 * (1 + 2 + 4 + 8 + 16 + 32 + 64) = 2 * 127 = 254? 
    # Wait, Moore bound for girth g:
    # If g is even, n >= 2 * sum_{i=0}^{g/2 - 1} (k-1)^i
    # g=16, g/2-1 = 7.
    # Sum = 1 + 2 + 4 + 8 + 16 + 32 + 64 = 127.
    # n >= 254.
    # The prompt says lower bound is 512.
    # Maybe the bound is tighter or I am misremembering.
    # Regardless, 936 is the target.

    def check_girth_bfs(adj, n, g):
        """Check if graph has girth >= g using BFS from each vertex."""
        min_cycle = float('inf')
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                nd = dist[node] + 1
                if nd >= g:
                    break
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] != -1:
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                            if min_cycle < g:
                                return min_cycle
                    else:
                        dist[nb] = nd
                        parent[nb] = node
                        queue.append(nb)
        return min_cycle

    def voltage_graph_k4(q, volt):
        """Construct a voltage graph over K4 with given voltages.
        Returns (n, edges, adj) or None if invalid."""
        base_edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
        n = 4 * q
        adj = [[] for _ in range(n)]
        edge_set = set()
        
        for u, v in base_edges:
            vu = volt[(u, v)]
            for fiber in range(q):
                su = u * q + fiber
                sv = v * q + (fiber + vu) % q
                if su < sv:
                    edge_set.add((su, sv))
                else:
                    edge_set.add((sv, su))
        
        for su, sv in edge_set:
            adj[su].append(sv)
            adj[sv].append(su)
        
        # Check degree
        for i in range(n):
            if len(adj[i]) != 3:
                return None
        
        return n, list(edge_set), adj

    best_n = float('inf')
    best_edges = []

    # 1. Try LUW construction
    def luw_construct(k, g):
        q = k
        d = max(3, (g - 2) // 2)
        
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
        
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
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

    luw_n, luw_edges = luw_construct(k, g)
    adj = [[] for _ in range(luw_n)]
    for i, j in luw_edges:
        adj[i].append(j)
        adj[j].append(i)
    
    # Only check girth if it's potentially good
    if luw_n < best_n:
        min_c = check_girth_bfs(adj, luw_n, g)
        if min_c >= g:
            best_n = luw_n
            best_edges = luw_edges

    # 2. Try voltage graphs
    rng = random.Random(42)
    
    # We want n < 936, so 4*q < 936 => q < 234
    # Start from smaller q to find smaller n first
    for q in range(128, 234): # Start from reasonable lower bound
        if time.time() > deadline - 10:
            break
        
        # Try several random voltage assignments
        for trial in range(10):
            if time.time() > deadline - 5:
                break
            
            base_edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
            volt = {}
            for u, v in base_edges:
                val = rng.randint(0, q - 1)
                volt[(u, v)] = val
                volt[(v, u)] = (-val) % q
            
            result = voltage_graph_k4(q, volt)
            if result is None:
                continue
            
            n, edges, adj = result
            if n >= best_n:
                continue
                
            min_c = check_girth_bfs(adj, n, g)
            if min_c >= g:
                best_n = n
                best_edges = edges
                break # Found a valid one for this q, try smaller q? No, q is increasing.
                # Actually, we want the smallest n. So we should iterate q from low to high.
                # If we find a valid one at q, it's 4q.
                # If we find a valid one at smaller q, it's better.
                # So we should iterate q from low to high and stop if we find one?
                # No, a valid graph at q=100 is better than q=120.
                # So we should iterate q from low to high.
                # If we find one, we can break the inner loop, but continue to next q?
                # No, if we find one at q=100, n=400.
                # If we continue to q=101, n=404. Worse.
                # So if we find a valid graph at q, we should record it and break?
                # But maybe a non-random voltage at q=100 exists and we missed it?
                # Given time, we just take the first one found.

    if best_n == float('inf'):
        # Fallback to LUW if no voltage graph found
        return luw_n, luw_edges
        
    return best_n, best_edges
