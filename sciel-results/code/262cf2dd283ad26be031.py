import time
import random
import sys
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

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

    # First, get the LUW construction as a baseline
    luw_n, luw_edges = luw_construct(k, g)
    
    # Verify LUW graph girth
    adj = [[] for _ in range(luw_n)]
    for i, j in luw_edges:
        adj[i].append(j)
        adj[j].append(i)
    min_c = check_girth_bfs(adj, luw_n, g)
    
    best_n = luw_n
    best_edges = luw_edges
    
    if min_c < g:
        # LUW didn't work, try larger d
        for d_try in range(8, 12):
            if time.time() > deadline - 30:
                break
            q = k
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
            for c in range(5, d_try + 1):
                r = (c - 5) % 4
                c0 = c - (c - 5) % 4
                rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

            def neighbours2(vertex, rule=rule, d=d_try, q=q, add=add, sub=sub, mul=mul):
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

            start2 = (0, (None,) + (0,) * d_try)
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
            for i, j in edges2:
                adj2[i].append(j)
                adj2[j].append(i)
            
            min_c2 = check_girth_bfs(adj2, n2, g)
            if min_c2 >= g:
                return n2, edges2
        
        return best_n, best_edges

    # Try voltage graph construction over K_4 to potentially find smaller graph
    # K_4 has 4 vertices and 6 edges, each vertex has degree 3
    # A voltage graph over K_4 with group Z_q gives a 3-regular graph with 4*q vertices
    # We need to find voltages such that girth >= 16
    
    # For girth 16, we need no cycles shorter than 16
    # In a voltage graph, cycles correspond to closed walks in the base graph
    # with product of voltages equal to identity
    
    # Try different q values and random voltage assignments
    rng = random.Random(42)
    
    for q in range(17, 250):
        if time.time() > deadline - 15:
            break
        if 4 * q >= best_n:
            break
        
        # Try several random voltage assignments
        for trial in range(15):
            if time.time() > deadline - 10:
                break
            
            # K_4 vertices: 0, 1, 2, 3
            # Edges: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
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
            min_c = check_girth_bfs(adj, n, g)
            if min_c >= g:
                if n < best_n:
                    best_n = n
                    best_edges = edges
                    break
    
    return best_n, best_edges
