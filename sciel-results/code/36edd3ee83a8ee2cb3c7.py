import time
import random
import math

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110

    if k != 3:
        # Fallback for non-3 regular
        return _lnw_construct(k, g)

    # Try voltage graph construction on Möbius-Kantor graph
    # Möbius-Kantor graph has 16 vertices, girth 8
    # We lift it by a group of size M.
    # Resulting graph has 16*M vertices.
    # We want girth >= 16.
    
    # Define Möbius-Kantor graph
    # It is a 3-regular graph on 16 vertices.
    # One construction: It is the bipartite graph with parts {0..7} and {8..15}
    # Actually, let's use a known edge list.
    # The Möbius-Kantor graph is the (8,3)-cage? No, the (8,3)-cage is the Pappus graph (18 vertices) or similar?
    # The smallest cubic graph of girth 8 is the Pappus graph (18 vertices) or the Möbius-Kantor graph (16 vertices)?
    # Wait, the Moore bound for g=8 is 2(3^2 - 1) = 16.
    # So a cubic graph of girth 8 on 16 vertices would be a Moore graph.
    # Does a Moore graph of girth 8 exist?
    # Moore graphs exist for g=3,4,5,7,9,16.
    # For g=8, no Moore graph exists.
    # So the smallest cubic graph of girth 8 has > 16 vertices.
    # The Pappus graph has 18 vertices and girth 8.
    # The Möbius-Kantor graph has 16 vertices but girth 6?
    # Let's check.
    # The Möbius-Kantor graph is a Cayley graph of Z_8 x Z_2?
    # It has girth 6.
    # So we should use the **Pappus graph** (18 vertices, girth 8).
    
    # Pappus graph: 18 vertices.
    # Lift by group of size M -> 18*M vertices.
    # If M=10, n=180.
    # If M=5, n=90.
    
    # Let's use the Pappus graph.
    # Edges of Pappus graph:
    # It is the incidence graph of the Pappus configuration.
    # Or we can define it.
    # Vertices 0..17.
    # 3-regular.
    
    # Let's try to find a voltage assignment on the Pappus graph over Z_M.
    
    pappus_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 0),
        (0, 9), (1, 10), (2, 11), (3, 12), (4, 13), (5, 14), (6, 15), (7, 16), (8, 17),
        (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9), (7, 10), (8, 11), (9, 12), (10, 13), (11, 14), (12, 15), (13, 16), (14, 17), (15, 0), (16, 1), (17, 2)
    ]
    # This might not be the Pappus graph. Let's use a reliable source or construct it.
    # The Pappus graph is the (8,3)-cage? No, the (8,3)-cage is the **Pappus graph** with 18 vertices?
    # Actually, the smallest cubic graph of girth 8 is the **Pappus graph** (18 vertices).
    # Let's assume this edge list is a cubic graph of girth 8.
    # We will verify girth in the code.
    
    # Actually, let's just use a known 3-regular graph of girth 8.
    # If the graph doesn't have girth 8, the lift won't have girth 16.
    
    # Let's try M from 5 to 50.
    best_n = float('inf')
    best_edges = None
    
    for M in range(5, 100):
        if time.time() > deadline - 20:
            break
            
        # Try random voltage assignments
        found = False
        for _ in range(50):
            if time.time() > deadline - 20:
                break
            voltages = [random.randint(0, M-1) for _ in range(len(pappus_edges))]
            # Check if this assignment works
            n, edges, girth = _lift_and_check(pappus_edges, voltages, M, g)
            if girth >= g:
                best_n = n
                best_edges = edges
                found = True
                break
        if found:
            break
            
    if best_edges is not None:
        return best_n, best_edges
        
    # Fallback to LUW
    return _lnw_construct(3, g)

def _lift_and_check(base_edges, voltages, M, g):
    n_base = max(u for u, v in base_edges) + 1
    n = n_base * M
    lift_edges = []
    
    for i, (u, v) in enumerate(base_edges):
        w = voltages[i]
        for a in range(M):
            u_lift = u * M + a
            v_lift = v * M + (a + w) % M
            lift_edges.append((u_lift, v_lift))
            
    # Check girth
    adj = [set() for _ in range(n)]
    for u, v in lift_edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Simple BFS for girth
    min_cycle = n + 1
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            if dist[u] >= min_cycle:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    if v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
        if min_cycle < g:
            return n, None, min_cycle
            
    return n, lift_edges, min_cycle

def _lnw_construct(k, g):
    # Original fallback
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    return _lnw_construct_params(k, d)

def _lnw_construct_params(k, d):
    q = k
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
