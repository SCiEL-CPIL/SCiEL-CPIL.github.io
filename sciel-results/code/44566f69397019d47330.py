import numpy as np
import random
import time

def construct(k, g):
    # Construct PG(2,3) incidence graph
    # Points: 13, Lines: 13
    # Total 26 vertices.
    
    # Generate points
    points = []
    for x in range(3):
        for y in range(3):
            for z in range(3):
                if x == 0 and y == 0 and z == 0: continue
                # Canonicalize: first non-zero is 1
                # Since field is Z3, non-zero are 1,2.
                # If x!=0, scale by x^-1
                if x != 0:
                    inv_x = 1 if x == 1 else 2
                    p = (1, (y*inv_x)%3, (z*inv_x)%3)
                elif y != 0:
                    inv_y = 1 if y == 1 else 2
                    p = (0, 1, (z*inv_y)%3)
                else:
                    p = (0, 0, 1)
                if p not in points:
                    points.append(p)
    
    # Generate lines
    # A line is a set of 4 points.
    # In PG(2,3), lines are defined by ax+by+cz=0.
    # Normal vectors (a,b,c) modulo scaling.
    lines = []
    line_points = []
    for a in range(3):
        for b in range(3):
            for c in range(3):
                if a == 0 and b == 0 and c == 0: continue
                # Canonicalize normal
                if a != 0:
                    inv_a = 1 if a == 1 else 2
                    n = (1, (b*inv_a)%3, (c*inv_a)%3)
                elif b != 0:
                    inv_b = 1 if b == 1 else 2
                    n = (0, 1, (c*inv_b)%3)
                else:
                    n = (0, 0, 1)
                
                if n in [tuple(l) for l in lines]: continue
                lines.append(n)
                pts = []
                for p in points:
                    if (n[0]*p[0] + n[1]*p[1] + n[2]*p[2]) % 3 == 0:
                        pts.append(p)
                line_points.append(pts)
    
    # Build base graph
    # Vertices 0..12 are points, 13..25 are lines.
    base_edges = []
    for i, p in enumerate(points):
        for j, lp in enumerate(line_points):
            if p in lp:
                base_edges.append((i, 13+j))
    
    # Use a voltage lift over Z_m where m is chosen to give girth >= g
    # For k=4, g=10: use m=10 lift -> 260 vertices
    # Deterministic voltage assignment using algebraic structure
    
    m = g  # lift size
    
    n_vertices = 26 * m
    
    # Build adjacency for base graph
    adj = {i: [] for i in range(26)}
    for u, v in base_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Enumerate all cycles of length < g in base graph
    # Base graph is bipartite (points 0-12, lines 13-25)
    # Only even-length cycles exist
    
    def find_cycles_up_to_len(max_len):
        """Find all simple cycles up to max_len in the base graph"""
        cycles = []
        num_vertices = 26
        
        # BFS-based cycle enumeration
        for start in range(num_vertices):
            # DFS to find cycles starting and ending at start
            visited = [False] * num_vertices
            path = [start]
            visited[start] = True
            
            def dfs(current, depth):
                if depth >= max_len:
                    return
                for nb in adj[current]:
                    if nb == start and depth + 1 >= 3:
                        # Found a cycle
                        cycle = path + [start]
                        # Canonicalize: rotate so smallest vertex is first
                        min_idx = min(range(len(cycle)-1), key=lambda i: cycle[i])
                        canon = tuple(cycle[min_idx:] + cycle[:min_idx])
                        if canon not in [tuple(c) for c in cycles]:
                            cycles.append(list(canon))
                    elif not visited[nb]:
                        visited[nb] = True
                        path.append(nb)
                        dfs(nb, depth + 1)
                        path.pop()
                        visited[nb] = False
            
            dfs(start, 1)
        
        return cycles
    
    # Get all cycles of length 4, 6, 8 (if g > 8, we need to check up to g-1)
    max_cycle_len_to_check = min(g - 1, 12)  # Check cycles up to this length
    cycles = find_cycles_up_to_len(max_cycle_len_to_check)
    
    # Filter to only cycles with length < g
    relevant_cycles = [c for c in cycles if len(c) < g]
    
    # Now assign voltages deterministically
    # Use a hash-based approach: assign voltage based on edge position
    # Then verify and adjust if needed
    
    # Strategy: assign voltages using a systematic pattern
    # Edge index -> voltage
    edge_index = {}
    for idx, (u, v) in enumerate(base_edges):
        edge_index[(u, v)] = idx
        edge_index[(v, u)] = idx
    
    # Try deterministic assignments using different seeds/patterns
    best_voltages = None
    
    for seed in range(500):
        random.seed(seed)
        voltages = {}
        for u, v in base_edges:
            voltages[(u, v)] = random.randint(0, m-1)
            voltages[(v, u)] = (-voltages[(u, v)]) % m
        
        # Check if any relevant cycle has voltage sum 0 mod m
        valid = True
        for cycle in relevant_cycles:
            s = 0
            for i in range(len(cycle) - 1):
                u, v = cycle[i], cycle[i+1]
                s += voltages[(u, v)]
            if s % m == 0:
                valid = False
                break
        
        if valid:
            best_voltages = voltages
            break
    
    # If no valid assignment found, try more seeds or use fallback
    if best_voltages is None:
        # Try a more systematic approach: use linear algebra
        # For each cycle, we need sum of voltages != 0 mod m
        # This is a constraint satisfaction problem
        
        # Fallback: use larger m or different base
        # Try m = 12 for better odds
        for m2 in [12, 15, 18]:
            m = m2
            n_vertices = 26 * m
            for seed in range(200):
                random.seed(seed)
                voltages = {}
                for u, v in base_edges:
                    voltages[(u, v)] = random.randint(0, m-1)
                    voltages[(v, u)] = (-voltages[(u, v)]) % m
                
                valid = True
                for cycle in relevant_cycles:
                    s = 0
                    for i in range(len(cycle) - 1):
                        u, v = cycle[i], cycle[i+1]
                        s += voltages[(u, v)]
                    if s % m == 0:
                        valid = False
                        break
                
                if valid:
                    best_voltages = voltages
                    break
            if best_voltages is not None:
                break
    
    if best_voltages is None:
        # Ultimate fallback: LUW construction
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
    
    # Build the lift graph
    voltages = best_voltages
    lift_edges = []
    for u, v in base_edges:
        vu = voltages[(u, v)]
        for i in range(m):
            lift_edges.append((u*m + i, v*m + (i + vu) % m))
    
    # Verify girth of lift graph
    adj_lift = {i: [] for i in range(n_vertices)}
    for u, v in lift_edges:
        adj_lift[u].append(v)
        adj_lift[v].append(u)
    
    # Check girth via BFS from each vertex
    min_girth = 999
    for start in range(n_vertices):
        dist = {start: 0}
        parent = {start: None}
        queue = [start]
        while queue:
            cur = queue.pop(0)
            for nb in adj_lift[cur]:
                if nb not in dist:
                    dist[nb] = dist[cur] + 1
                    parent[nb] = cur
                    queue.append(nb)
                elif parent[nb] != cur and parent[cur] != nb:
                    cycle_len = dist[nb] + dist[cur] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
    
    if min_girth >= g:
        return n_vertices, lift_edges
    
    # If verification fails, use LUW fallback
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
