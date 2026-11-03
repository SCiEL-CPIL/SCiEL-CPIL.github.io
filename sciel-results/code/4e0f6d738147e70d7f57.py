import numpy as np
import random
import time

def construct(k, g):
    # Construct PG(2,3) incidence graph
    points = []
    for x in range(3):
        for y in range(3):
            for z in range(3):
                if x == 0 and y == 0 and z == 0: continue
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
    
    lines = []
    line_points = []
    for a in range(3):
        for b in range(3):
            for c in range(3):
                if a == 0 and b == 0 and c == 0: continue
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
    
    base_edges = []
    adj = {i: [] for i in range(26)}
    for i, p in enumerate(points):
        for j, lp in enumerate(line_points):
            if p in lp:
                base_edges.append((i, 13+j))
                adj[i].append(13+j)
                adj[13+j].append(i)
                
    # Enumerate 6-cycles
    cycles_6 = []
    for p1 in range(13):
        for p2 in range(13):
            if p2 == p1: continue
            for p3 in range(13):
                if p3 == p1 or p3 == p2: continue
                L12 = -1
                for l in range(13):
                    if points[p1] in line_points[l] and points[p2] in line_points[l]:
                        L12 = 13 + l
                        break
                L23 = -1
                for l in range(13):
                    if points[p2] in line_points[l] and points[p3] in line_points[l]:
                        L23 = 13 + l
                        break
                L31 = -1
                for l in range(13):
                    if points[p3] in line_points[l] and points[p1] in line_points[l]:
                        L31 = 13 + l
                        break
                        
                if L12 == -1 or L23 == -1 or L31 == -1: continue
                if L12 == L23 or L23 == L31 or L31 == L12: continue
                
                cycle_edges = [
                    (p1, L12), (L12, p2), (p2, L23),
                    (L23, p3), (p3, L31), (L31, p1)
                ]
                cycles_6.append(cycle_edges)
                
    unique_cycles_6 = set()
    for ce in cycles_6:
        key = frozenset(tuple(sorted(e)) for e in ce)
        unique_cycles_6.add(key)
    cycles_6 = [list(c) for c in unique_cycles_6]
    
    # Enumerate 8-cycles
    cycles_8 = []
    for p1 in range(13):
        for l1 in adj[p1]:
            for p2 in adj[l1]:
                if p2 == p1: continue
                for l2 in adj[p2]:
                    if l2 == l1: continue
                    for p3 in adj[l2]:
                        if p3 == p1 or p3 == p2: continue
                        for l3 in adj[p3]:
                            if l3 == l2: continue
                            for p4 in adj[l3]:
                                if p4 in [p1, p2, p3]: continue
                                for l4 in adj[p4]:
                                    if l4 == l3: continue
                                    if p1 in adj[l4]:
                                        cycle_edges = [
                                            (p1, l1), (l1, p2), (p2, l2),
                                            (l2, p3), (p3, l3), (l3, p4),
                                            (p4, l4), (l4, p1)
                                        ]
                                        cycles_8.append(cycle_edges)
                                        
    unique_cycles_8 = set()
    for ce in cycles_8:
        key = frozenset(tuple(sorted(e)) for e in ce)
        unique_cycles_8.add(key)
    cycles_8 = [list(c) for c in unique_cycles_8]
    
    all_cycles = cycles_6 + cycles_8
    
    m = 10
    n_vertices = 26 * m
    
    best_voltages = None
    for attempt in range(20):
        voltages = {}
        for u, v in base_edges:
            voltages[(u, v)] = random.randint(0, m-1)
            voltages[(v, u)] = (-voltages[(u, v)]) % m
            
        for step in range(2000):
            violated = None
            for ce in all_cycles:
                s = 0
                for u, v in ce:
                    s += voltages[(u, v)]
                if s % m == 0:
                    violated = ce
                    break
                    
            if not violated:
                best_voltages = voltages
                break
                
            u, v = random.choice(violated)
            old_v = voltages[(u, v)]
            new_v = random.randint(0, m-1)
            while new_v == old_v:
                new_v = random.randint(0, m-1)
            voltages[(u, v)] = new_v
            voltages[(v, u)] = (-new_v) % m
            
        if best_voltages:
            break
            
    if best_voltages:
        voltages = best_voltages
        lift_edges = []
        for u, v in base_edges:
            vu = voltages[(u, v)]
            for i in range(m):
                lift_edges.append((u*m + i, v*m + (i + vu) % m))
        return n_vertices, lift_edges
    else:
        # Fallback to LUW if search fails
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
