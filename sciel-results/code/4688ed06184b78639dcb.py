import numpy as np
import random
import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    if k != 4 or g != 10:
        # Fallback for other parameters: use the seed's generic construction
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

    # For k=4, g=10: try to construct a 320-vertex cage
    # The (4,10)-cage is known to have 320 vertices.
    # One known construction uses the incidence graph of the generalized hexagon G(3,3).
    
    # Let's try a voltage lift approach on a small base graph.
    # The base graph should be 4-regular with small girth.
    # Petersen graph is 3-regular on 10 vertices, girth 5.
    # We need a 4-regular base.
    
    # Let's try a different approach: use a known 320-vertex graph.
    # The (4,10)-cage is the incidence graph of G(3,3).
    
    # G(3,3) has 112 points and 112 lines, so the incidence graph has 224 vertices?
    # No, G(3,3) has 112 points and 112 lines, but the incidence graph would have
    # 112+112 = 224 vertices, each with degree 4 (since each point is on 4 lines
    # and each line has 4 points in G(3,3)). But 224 < 320.
    
    # Actually, let me reconsider. The generalized hexagon of order (3,3) has:
    # Number of points = (s+1)(s^2+1) = 4*10 = 40? No...
    # For a generalized hexagon of order (s,t):
    # Number of points = (s+1)(s^2 + s + 1)... hmm, let me just use a known construction.
    
    # Let's use a different strategy: try to find voltages on a 26-vertex base graph
    # (PG(2,3) incidence graph) with m=13, giving 338 vertices, but verify carefully.
    
    # Actually, let's just try to construct a valid 4-regular graph with girth >= 10
    # using a random approach with local search.
    
    # First, try the PG(2,3) lift with m=13 (338 vertices)
    # This is known to have girth 10.
    
    points = []
    for x in range(3):
        for y in range(3):
            for z in range(3):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = 1 if x == 1 else 2
                    p = (1, (y * inv_x) % 3, (z * inv_x) % 3)
                elif y != 0:
                    inv_y = 1 if y == 1 else 2
                    p = (0, 1, (z * inv_y) % 3)
                else:
                    p = (0, 0, 1)
                if p not in points:
                    points.append(p)
    
    lines = []
    line_points = []
    for a in range(3):
        for b in range(3):
            for c in range(3):
                if a == 0 and b == 0 and c == 0:
                    continue
                if a != 0:
                    inv_a = 1 if a == 1 else 2
                    n = (1, (b * inv_a) % 3, (c * inv_a) % 3)
                elif b != 0:
                    inv_b = 1 if b == 1 else 2
                    n = (0, 1, (c * inv_b) % 3)
                else:
                    n = (0, 0, 1)
                if n in [tuple(l) for l in lines]:
                    continue
                lines.append(n)
                pts = []
                for p in points:
                    if (n[0] * p[0] + n[1] * p[1] + n[2] * p[2]) % 3 == 0:
                        pts.append(p)
                line_points.append(pts)
    
    base_edges = []
    for i, p in enumerate(points):
        for j, lp in enumerate(line_points):
            if p in lp:
                base_edges.append((i, 13 + j))
    
    adj_base = {i: [] for i in range(26)}
    for u, v in base_edges:
        adj_base[u].append(v)
        adj_base[v].append(u)
    
    m = 13
    n_vertices = 26 * m
    
    # Try to find valid voltages
    random.seed(42)
    found = False
    best_edges = None
    
    # Precompute all 6-cycles in the base graph for faster checking
    six_cycles = []
    for p1 in range(13):
        for l1 in adj_base[p1]:
            for p2 in adj_base[l1]:
                if p2 == p1:
                    continue
                for l2 in adj_base[p2]:
                    if l2 == l1:
                        continue
                    for p3 in adj_base[l2]:
                        if p3 == p1 or p3 == p3 and p3 == p2:
                            continue
                        if p3 == p1 or p3 == p2:
                            continue
                        for l3 in adj_base[p3]:
                            if l3 == l2:
                                continue
                            if p1 in adj_base[l3]:
                                cycle = [(p1, l1), (l1, p2), (p2, l2), (l2, p3), (p3, l3), (l3, p1)]
                                six_cycles.append(cycle)
    
    # Also precompute 8-cycles
    eight_cycles = []
    for p1 in range(13):
        for l1 in adj_base[p1]:
            for p2 in adj_base[l1]:
                if p2 == p1:
                    continue
                for l2 in adj_base[p2]:
                    if l2 == l1:
                        continue
                    for p3 in adj_base[l2]:
                        if p3 == p1 or p3 == p2:
                            continue
                        for l3 in adj_base[p3]:
                            if l3 == l2:
                                continue
                            if p1 in adj_base[l3]:
                                continue  # This would be a 6-cycle
                            for p4 in adj_base[l3]:
                                if p4 in (p1, p2, p3):
                                    continue
                                for l4 in adj_base[p4]:
                                    if l4 == l3:
                                        continue
                                    if p1 in adj_base[l4]:
                                        cycle = [(p1, l1), (l1, p2), (p2, l2), (l2, p3), (p3, l3), (l3, p4), (p4, l4), (l4, p1)]
                                        eight_cycles.append(cycle)
    
    for attempt in range(2000):
        if time.time() > deadline - 10:
            break
        
        voltages = {}
        for u, v in base_edges:
            voltages[(u, v)] = random.randint(0, m - 1)
            voltages[(v, u)] = (-voltages[(u, v)]) % m
        
        valid = True
        # Check 6-cycles
        for cycle in six_cycles:
            s = 0
            for (a, b) in cycle:
                s += voltages[(a, b)]
            if s % m == 0:
                valid = False
                break
        
        if not valid:
            continue
        
        # Check 8-cycles
        for cycle in eight_cycles:
            s = 0
            for (a, b) in cycle:
                s += voltages[(a, b)]
            if s % m == 0:
                valid = False
                break
        
        if not valid:
            continue
        
        # Build lift and verify girth exactly
        lift_edges = []
        for u, v in base_edges:
            vu = voltages[(u, v)]
            for i in range(m):
                lift_edges.append((u * m + i, v * m + (i + vu) % m))
        
        # Verify girth using BFS from each vertex
        adj_lift = [[] for _ in range(n_vertices)]
        for u, v in lift_edges:
            adj_lift[u].append(v)
            adj_lift[v].append(u)
        
        min_girth = 999
        for start in range(n_vertices):
            dist = {start: 0}
            parent = {start: None}
            queue = [start]
            head = 0
            while head < len(queue):
                cur = queue[head]
                head += 1
                for nb in adj_lift[cur]:
                    if nb not in dist:
                        dist[nb] = dist[cur] + 1
                        parent[nb] = cur
                        queue.append(nb)
                    elif parent[nb] != cur and parent[cur] != nb:
                        cycle_len = dist[nb] + dist[cur] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                        if min_girth < g:
                            break
                if min_girth < g:
                    break
            if min_girth < g:
                break
        
        if min_girth >= g:
            return n_vertices, lift_edges
    
    # If we didn't find a 338-vertex graph, use the fallback
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
