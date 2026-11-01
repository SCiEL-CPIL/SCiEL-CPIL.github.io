import time
import random
import numpy as np

def construct(k, g):
    """
    Constructs a k-regular graph of girth at least g.
    Uses voltage assignment on K_{4,4} for k=4.
    """
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    if k != 4:
        return _seed_construct(k, g)
        
    # Base graph: K_{4,4}
    # Vertices: 0..3 (left), 4..7 (right)
    # Edges: (i, 4+j) for all i, j
    base_vertices = 8
    base_edges = []
    for i in range(4):
        for j in range(4):
            base_edges.append((i, 4 + j))
            
    best_n = 320
    best_edges = None
    
    # Iterate m from small to large to find smallest n first
    for m in range(31, 40):
        if time.time() > deadline:
            break
            
        found = False
        for attempt in range(5000):
            if time.time() > deadline:
                break
                
            voltages = {}
            for u, v in base_edges:
                val = random.randint(1, m - 1)
                voltages[(u, v)] = val
                voltages[(v, u)] = (-val) % m
            
            valid = _check_voltage_girth(voltages, m, base_edges)
            if valid:
                n = base_vertices * m
                edges = []
                for u, v in base_edges:
                    val = voltages[(u, v)]
                    for g_elem in range(m):
                        u_new = u * m + g_elem
                        v_new = v * m + (g_elem + val) % m
                        edges.append((u_new, v_new))
                
                if _check_girth(n, edges, g):
                    best_n = n
                    best_edges = edges
                    found = True
                    break
        
        if found:
            break

    if best_edges is not None:
        return best_n, best_edges
        
    return _seed_construct(k, g)

def _check_voltage_girth(voltages, m, base_edges):
    """Check if voltage assignment avoids cycles of length < 10 in base graph."""
    # Precompute voltage lookup
    # Left vertices: 0-3, Right vertices: 4-7
    
    # Check 4-cycles: u1-v1-u2-v2-u1 where u1!=u2, v1!=v2
    for u1 in range(4):
        for u2 in range(u1 + 1, 4):
            for v1 in range(4):
                for v2 in range(v1 + 1, 4):
                    s = (voltages[(u1, 4+v1)] + 
                         voltages[(4+v1, u2)] + 
                         voltages[(u2, 4+v2)] + 
                         voltages[(4+v2, u1)]) % m
                    if s == 0:
                        return False
    
    # Check 6-cycles: u1-v1-u2-v2-u3-v3-u1
    # u1,u2,u3 distinct left; v1,v2,v3 distinct right
    for u1 in range(4):
        for u2 in range(4):
            if u2 == u1: continue
            for u3 in range(4):
                if u3 == u1 or u3 == u2: continue
                for v1 in range(4):
                    for v2 in range(4):
                        if v2 == v1: continue
                        for v3 in range(4):
                            if v3 == v1 or v3 == v2: continue
                            s = (voltages[(u1, 4+v1)] + 
                                 voltages[(4+v1, u2)] + 
                                 voltages[(u2, 4+v2)] + 
                                 voltages[(4+v2, u3)] + 
                                 voltages[(4+v3, u1)]) % m
                            if s == 0:
                                return False
    
    # Check 8-cycles: u1-v1-u2-v2-u3-v3-u4-v4-u1
    # All u's distinct, all v's distinct
    left = [0, 1, 2, 3]
    right = [0, 1, 2, 3]
    for i1 in range(4):
        for i2 in range(4):
            if i2 == i1: continue
            for i3 in range(4):
                if i3 == i1 or i3 == i2: continue
                for i4 in range(4):
                    if i4 in (i1, i2, i3): continue
                    u1, u2, u3, u4 = left[i1], left[i2], left[i3], left[i4]
                    for j1 in range(4):
                        for j2 in range(4):
                            if j2 == j1: continue
                            for j3 in range(4):
                                if j3 == j1 or j3 == j2: continue
                                for j4 in range(4):
                                    if j4 in (j1, j2, j3): continue
                                    v1, v2, v3, v4 = right[j1], right[j2], right[j3], right[j4]
                                    s = (voltages[(u1, 4+v1)] + 
                                         voltages[(4+v1, u2)] + 
                                         voltages[(u2, 4+v2)] + 
                                         voltages[(4+v2, u3)] + 
                                         voltages[(u3, 4+v3)] + 
                                         voltages[(4+v3, u4)] + 
                                         voltages[(u4, 4+v4)] + 
                                         voltages[(4+v4, u1)]) % m
                                    if s == 0:
                                        return False
    
    return True

def _check_girth(n, edges, g):
    """Check if graph has girth >= g using non-backtracking walks."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # For each vertex, do BFS tracking parent to detect shortest cycle
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _seed_construct(k, g):
    # Copy of the seed code
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
